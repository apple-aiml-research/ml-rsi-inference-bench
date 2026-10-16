#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Read-only receipt at ZMQ's actual receive/deserialization boundary.

Only already-deserialized public request IDs are read. No extra unpickle,
candidate queue/state methods, log messages, or model-owned fields are used.
"""
import functools
import hashlib
import threading


def install(emit):
    import pickle
    import zmq
    if getattr(pickle.loads, '_infb_request_receipt', False):
        return
    local = threading.local()
    receive, raw_receive, loads = zmq.Socket.recv_multipart, zmq.Socket.recv, pickle.loads

    @functools.wraps(receive)
    def recv(self, *args, **kwargs):
        frames = receive(self, *args, **kwargs)

        local.payload = frames[-1] if len(frames) >= 3 and frames[-2] == b'' else None
        return frames

    @functools.wraps(raw_receive)
    def recv_raw(self, *args, **kwargs):
        value = raw_receive(self, *args, **kwargs)
        local.payload = value if isinstance(value, bytes) else None
        return value

    @functools.wraps(loads)
    def deserialize(data, *args, **kwargs):
        value = loads(data, *args, **kwargs)
        expected = getattr(local, 'payload', None)
        if expected is not None and data == expected:
            local.payload = None
            try:
                for obj in value if isinstance(value, list) else [value]:
                    cls = type(obj)
                    if not cls.__module__.startswith('sglang.multimodal_gen.'):
                        continue
                    role = 'abort' if cls.__name__ == 'AbortReq' else 'generation' if cls.__name__ == 'Req' else None
                    if role and getattr(obj, 'request_id', None) is not None:
                        emit({'event': 'request_wire_received', 'role': role,
                              'request_id': str(obj.request_id), 'payload_sha256': hashlib.sha256(data).hexdigest()})
            except Exception as exc:
                emit({'event': 'request_wire_error', 'error': str(exc)[:250]})
        return value

    deserialize._infb_request_receipt = True
    zmq.Socket.recv_multipart, zmq.Socket.recv, pickle.loads = recv, recv_raw, deserialize
    emit({'event': 'request_wire_installed'})
