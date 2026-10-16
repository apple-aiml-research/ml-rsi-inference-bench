//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Pier and Harbor projects
// Vendored from https://github.com/datacurve-ai/pier at 3f688a1.
/**
 * Compact markdown renderer used inside the trajectory viewer.
 *
 * Pier's `~/components/ui/markdown.tsx` wraps everything in a card; we
 * want inline-flowing prose that sits naturally inside an assistant
 * turn, so we reimplement the bits we need with shadcn / tailwind tokens
 * and delegate fenced code blocks to pier's existing `CodeBlock`.
 */
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { CodeBlock } from "~/components/ui/code-block";

export function Markdown({ text }: { text: string }) {
  return (
    <div className="prose prose-sm max-w-none dark:prose-invert text-sm leading-relaxed prose-p:my-1.5 prose-headings:mt-3 prose-headings:mb-1.5 prose-headings:font-semibold prose-h1:text-[1.05em] prose-h2:text-[1em] prose-h3:text-[0.95em] prose-h4:text-[0.9em] prose-h5:text-[0.85em] prose-h6:text-[0.85em] prose-pre:my-2 prose-pre:p-0 prose-pre:bg-transparent prose-li:my-0 prose-ol:my-1.5 prose-ul:my-1.5 prose-blockquote:my-2 prose-blockquote:not-italic prose-blockquote:font-normal prose-blockquote:text-muted-foreground prose-a:text-primary prose-a:underline-offset-2">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          code: (props) => {
            const { className, children } = props as {
              className?: string;
              children?: React.ReactNode;
            };
            const match = /language-(\w+)/.exec(className ?? "");
            const code = String(children ?? "").replace(/\n$/, "");
            const isInline = !match && !code.includes("\n");
            if (isInline) {
              return (
                <code className="rounded-sm bg-muted px-[0.3em] py-[0.1em] font-mono text-[0.86em]">
                  {children}
                </code>
              );
            }
            return <CodeBlock code={code} lang={match?.[1] ?? "text"} wrap />;
          },
          pre: ({ children }) => <>{children}</>,
          // Render links as inert text. Trajectory text is untrusted: an
          // agent's free-form output can include autolinked URLs whose host
          // resolves to a real TLD (e.g. .md, .sh, .zip) and clicking them
          // could exfiltrate state or pull surprising content. Show the URL
          // so the reader still sees what the model wrote, but disable
          // navigation and underline as a hover hint instead of a chrome.
          a: ({ children, href, title }) => (
            <span
              className="underline decoration-dotted decoration-muted-foreground/60 underline-offset-2"
              title={title ?? href ?? undefined}
            >
              {children}
            </span>
          ),
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}
