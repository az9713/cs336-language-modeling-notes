#!/usr/bin/env python3
"""Check inline and display MathJax layout in a generated HTML reader."""

import argparse
import sys
from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from urllib.parse import quote
from pathlib import Path

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

MEASURE_LAYOUT = """
() => {
    const formulas = [...document.querySelectorAll(
        'mjx-container, .katex, math:not(mjx-container math, .katex math, mjx-assistive-mml math)'
    )];
    const wrapperSelector = 'span.math, .inline-math, .math-inline, .equation, .math-display, .math-scroll, .katex-display';

    const boxes = new Map();
    let escapedMath = 0;
    for (const formula of formulas) {
        const inline = !['true', 'block'].includes(formula.getAttribute('display'))
            && !formula.closest('.katex-display');
        boxes.set(formula, inline);
        const wrapper = formula.parentElement.closest(wrapperSelector + (inline ? '' : ', div.math'));
        if (wrapper) boxes.set(wrapper, inline);
        const rect = formula.getBoundingClientRect();
        if (rect.width && rect.right > innerWidth + 1) {
            let enclosing = formula.parentElement;
            while (enclosing && enclosing !== document.body) {
                const overflow = getComputedStyle(enclosing).overflowX;
                if (['auto', 'scroll'].includes(overflow) && enclosing.scrollWidth > enclosing.clientWidth) {
                    break;
                }
                enclosing = enclosing.parentElement;
            }
            if (!enclosing || enclosing === document.body) escapedMath++;
        }
    }

    let unhiddenInlineScrollers = 0;
    let wideInline = 0;
    let wideDisplay = 0;
    let blockedOverflow = 0;
    for (const [box, inline] of boxes) {
        if (!box.clientWidth) continue;
        const style = getComputedStyle(box);
        const scrolls = ['auto', 'scroll'].includes(style.overflowX);
        const hidden = style.scrollbarWidth === 'none'
            || getComputedStyle(box, '::-webkit-scrollbar').display === 'none';
        const scrollsVertically = ['auto', 'scroll'].includes(style.overflowY);
        if (inline && !hidden && (scrolls || scrollsVertically)) {
            unhiddenInlineScrollers++;
        }
        if (box.scrollWidth - box.clientWidth <= 1) continue;
        // Overflow can reach an outer equation wrapper when this box leaves it visible.
        const outer = box.parentElement.closest(wrapperSelector + (inline ? '' : ', div.math'));
        if (style.overflowX === 'visible' && boxes.has(outer)) continue;
        if (inline) wideInline++;
        else wideDisplay++;
        const original = box.scrollLeft;
        box.scrollLeft = 24;
        if (box.scrollLeft === original) box.scrollLeft = -24;
        if (!scrolls || box.scrollLeft === original) blockedOverflow++;
        box.scrollLeft = original;
    }
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    const renderer = document.querySelector('script[src*="mathjax" i], script[src*="katex" i]');
    let unrendered = false;
    if (renderer || window.MathJax?.tex?.inlineMath) {
        while (walker.nextNode()) {
            const node = walker.currentNode;
            if (node.parentElement.closest('pre, code, script, style, textarea, mjx-container, .katex')) continue;
            if (node.textContent.includes(String.fromCharCode(92) + '(')
                || node.textContent.includes(String.fromCharCode(92) + '[')
                || node.textContent.includes('$$')) {
                unrendered = true;
                break;
            }
        }
    }
    return {
        unrendered,
        rendered: formulas.length,
        mathErrors: document.querySelectorAll('mjx-merror, .katex-error').length,
        unhiddenInlineScrollers, wideInline, wideDisplay, blockedOverflow, escapedMath,
        pageOverflow: document.documentElement.scrollWidth > innerWidth + 1,
    };
}
"""


def check_page(page, target):
    response = page.goto(target, wait_until="load")
    if response and not response.ok:
        raise RuntimeError(f"HTTP {response.status}")
    page.evaluate("""async () => {
        if (window.MathJax?.startup?.promise) await MathJax.startup.promise;
        else if (window.MathJax?.Hub) await new Promise(resolve => MathJax.Hub.Queue(resolve));
        await document.fonts.ready;
    }""")
    return page.evaluate(MEASURE_LAYOUT)


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


@contextmanager
def staged_site(directory, patterns):
    directory = directory.resolve(strict=True)
    files = sorted({path for pattern in patterns.splitlines() if pattern.strip()
                    for path in directory.glob(pattern.strip()) if path.is_file()})
    if not files:
        raise ValueError(f"No math pages match {patterns!r} in {directory}")
    handler = partial(QuietHandler, directory=str(directory))
    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}/"
            yield [base + quote(path.relative_to(directory).as_posix()) for path in files]
        finally:
            server.shutdown()
            thread.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pages", nargs="*", help="Local HTML files or deployed HTTP(S) URLs")
    parser.add_argument("--chrome", help="Path to an installed Chromium or Edge executable")
    parser.add_argument("--allow-no-math", action="store_true", help="Allow a site or sample with no equations")
    parser.add_argument("--site", type=Path, help="Staged static site to serve over loopback")
    parser.add_argument("--globs", help="Newline-separated page globs relative to --site")
    args = parser.parse_args()
    if args.site:
        if args.pages or not args.globs:
            parser.error("--site requires --globs and no positional pages")
        with staged_site(args.site, args.globs) as targets:
            return check_targets(targets, args)
    if not args.pages or args.globs:
        parser.error("provide pages or use --site with --globs")
    targets = [
        name if name.startswith(("http://", "https://"))
        else Path(name).resolve(strict=True).as_uri()
        for name in args.pages
    ]
    return check_targets(targets, args)


def check_targets(targets, args):
    failures = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=args.chrome,
            ignore_default_args=["--hide-scrollbars"],
        )
        for width, height in ((1440, 900), (390, 844)):
            page = browser.new_page(viewport={"width": width, "height": height})
            rendered_total = 0
            for target in targets:
                try:
                    result = check_page(page, target)
                except (RuntimeError, PlaywrightError) as error:
                    failures.append(f"{target} at {width}px: {error}")
                    continue
                rendered_total += result["rendered"]
                print(f"{target} at {width}px: {result}")
                if (
                    result["unrendered"]
                    or result["mathErrors"]
                    or result["unhiddenInlineScrollers"]
                    or result["blockedOverflow"]
                    or result["escapedMath"]
                ):
                    failures.append(f"{target} at {width}px: {result}")
            if not args.allow_no_math and not rendered_total:
                failures.append(f"No equations rendered at {width}px across {len(targets)} page(s)")
            page.close()
        browser.close()

    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"Passed {len(targets) * 2} desktop and mobile math layout checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
