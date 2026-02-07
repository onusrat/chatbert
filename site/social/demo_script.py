#!/usr/bin/env python3
"""ChatBERT-IMR Iterative Unmasking Demo Visualization.

Creates a compelling terminal animation of ChatBERT-IMR's iterative
unmasking process. Designed for recording demo GIFs/videos.

No model required -- uses hardcoded example for demonstration purposes.
Run: python demo_script.py

"""

import sys
import time
import os

# --- ANSI escape codes -------------------------------------------------------

RESET       = "\033[0m"
BOLD        = "\033[1m"
DIM         = "\033[2m"
ITALIC      = "\033[3m"

# Foreground colors
WHITE       = "\033[37m"

# Bright foreground
BR_BLACK    = "\033[90m"
BR_RED      = "\033[91m"
BR_GREEN    = "\033[92m"
BR_YELLOW   = "\033[93m"
BR_BLUE     = "\033[94m"
BR_MAGENTA  = "\033[95m"
BR_CYAN     = "\033[96m"
BR_WHITE    = "\033[97m"

# Background colors
BG_BLACK    = "\033[40m"

# Cursor control
CLEAR_SCREEN = "\033[2J"
HOME         = "\033[H"
HIDE_CURSOR  = "\033[?25l"
SHOW_CURSOR  = "\033[?25h"


# --- Demo data ----------------------------------------------------------------

PROMPT = "I just got a new puppy!"

# The final response tokens (based on actual ChatBERT output)
FINAL_TOKENS = [
    "haha", ",", "dogs", "are", "fun", "!",
    "what", "kind", "of", "puppy", "did",
    "you", "get", "?",
]

# Iterative unmasking schedule: each step is a list of
# (token_index, confidence_score) pairs for tokens revealed that step.
# Tokens are NOT revealed left-to-right -- high-confidence structural
# tokens and punctuation come first, then content words fill in.
UNMASK_SCHEDULE = [
    # Step 1: Punctuation and high-frequency tokens revealed first
    [(5, 0.97), (13, 0.96), (1, 0.95)],
    # Step 2: Function words the model is confident about
    [(3, 0.93), (8, 0.91), (11, 0.90)],
    # Step 3: Core content -- "dogs" and "fun" snap into place
    [(2, 0.88), (4, 0.86), (6, 0.84)],
    # Step 4: More content words fill in
    [(0, 0.82), (9, 0.79), (12, 0.77)],
    # Step 5: Final tokens -- "kind" and "did"
    [(7, 0.74), (10, 0.71)],
]

MASK_DISPLAY = "[MASK]"


# --- Rendering helpers --------------------------------------------------------

def render_token(token, state, is_new=False):
    """Render a single token with appropriate styling."""
    if state == "masked":
        return f"{DIM}{BR_BLACK}{MASK_DISPLAY}{RESET}"
    elif is_new:
        return f"{BOLD}{BR_GREEN}{BG_BLACK}{token}{RESET}"
    else:
        return f"{WHITE}{token}{RESET}"


def render_confidence_bar(score, width=25):
    """Render a confidence score as a colored bar."""
    filled = int(score * width)
    empty = width - filled

    if score >= 0.85:
        color = BR_GREEN
    elif score >= 0.70:
        color = BR_YELLOW
    else:
        color = BR_RED

    bar = f"{color}{'#' * filled}{DIM}{'.' * empty}{RESET}"
    return f"{bar} {color}{score:.0%}{RESET}"


def render_header():
    """Render the demo header."""
    lines = []
    lines.append(f"{BOLD}{BR_CYAN}{'=' * 72}{RESET}")
    lines.append(
        f"{BOLD}{BR_CYAN}  ChatBERT-IMR  "
        f"{DIM}{WHITE}Iterative Masked Language Model Refinement{RESET}"
    )
    lines.append(f"{DIM}{WHITE}  Deliberative Generation Demo{RESET}")
    lines.append(f"{BOLD}{BR_CYAN}{'=' * 72}{RESET}")
    return "\n".join(lines)


def render_prompt():
    """Render the user prompt."""
    lines = []
    lines.append("")
    lines.append(f"  {BOLD}{BR_BLUE}User:{RESET}  {WHITE}{PROMPT}{RESET}")
    lines.append("")
    lines.append(
        f"  {BOLD}{BR_MAGENTA}ChatBERT-IMR:{RESET}  "
        f"{DIM}{ITALIC}generating via iterative unmasking...{RESET}"
    )
    lines.append("")
    return "\n".join(lines)


def render_step_header(step, total_steps, revealed_count, total_count):
    """Render the step progress indicator."""
    progress = revealed_count / total_count
    bar_width = 30
    filled = int(progress * bar_width)
    empty = bar_width - filled

    step_str = f"  {BOLD}{BR_YELLOW}Iteration {step}/{total_steps}{RESET}"
    count_str = f"{WHITE}{revealed_count}/{total_count} tokens revealed{RESET}"
    bar = f"{BR_GREEN}{'|' * filled}{DIM}{BR_BLACK}{'.' * empty}{RESET}"

    return f"{step_str}  [{bar}]  {count_str}"


def render_tokens_multiline(tokens, states, new_indices=None, tokens_per_line=8):
    """Render tokens wrapped across multiple lines."""
    if new_indices is None:
        new_indices = set()

    lines = []
    current_line = "    "
    count = 0

    for i, (token, state) in enumerate(zip(tokens, states)):
        is_new = i in new_indices
        rendered = render_token(token, state, is_new)

        if token in {".", ",", "'s", "!"}:
            spacer = ""
        else:
            spacer = " " if count > 0 else ""

        current_line += spacer + rendered
        count += 1

        if count >= tokens_per_line and i < len(tokens) - 1:
            lines.append(current_line)
            current_line = "    "
            count = 0

    if current_line.strip():
        lines.append(current_line)

    return "\n".join(lines)


def render_confidence_panel(step_data, tokens):
    """Render confidence scores for newly unmasked tokens."""
    lines = []
    lines.append(
        f"\n  {DIM}{WHITE}Confidence scores for newly revealed tokens:{RESET}"
    )

    for idx, conf in sorted(step_data, key=lambda x: -x[1]):
        token = tokens[idx]
        bar = render_confidence_bar(conf, width=25)
        pos_str = f"{DIM}pos {idx:>2}{RESET}"
        tok_str = f"{BOLD}{BR_GREEN}{token:>14}{RESET}"
        lines.append(f"    {pos_str}  {tok_str}  {bar}")

    return "\n".join(lines)


def build_final_response():
    """Build the final response as a clean string."""
    parts = []
    for i, token in enumerate(FINAL_TOKENS):
        if token in {".", ",", "'s", "!"}:
            parts.append(token)
        else:
            if parts:
                parts.append(" " + token)
            else:
                parts.append(token)
    return "".join(parts)


def render_footer():
    """Render the final footer."""
    response = build_final_response()
    lines = []
    lines.append("")
    lines.append(f"  {BOLD}{BR_CYAN}{'-' * 72}{RESET}")
    lines.append(
        f"  {BOLD}{BR_MAGENTA}ChatBERT-IMR:{RESET}  {WHITE}{response}{RESET}"
    )
    lines.append("")
    lines.append(
        f"  {DIM}{WHITE}Generated in 5 iterations | {len(FINAL_TOKENS)} tokens | "
        f"~80ms total | 66M parameters{RESET}"
    )
    lines.append(
        f"  {DIM}{WHITE}Model: DistilBERT backbone | "
        f"Mask schedule: confidence-based{RESET}"
    )
    lines.append(f"  {BOLD}{BR_CYAN}{'=' * 72}{RESET}")
    return "\n".join(lines)


# --- Animation engine ---------------------------------------------------------

def draw_frame(content):
    """Draw a full frame to the terminal."""
    sys.stdout.write(HOME)
    sys.stdout.write(content)
    sys.stdout.flush()


def run_demo():
    """Run the full iterative unmasking demo animation."""
    total_steps = len(UNMASK_SCHEDULE)
    total_tokens = len(FINAL_TOKENS)

    # Track state of each token
    states = ["masked"] * total_tokens
    revealed_count = 0

    try:
        sys.stdout.write(HIDE_CURSOR)
        sys.stdout.write(CLEAR_SCREEN + HOME)
        sys.stdout.flush()

        header = render_header()
        prompt = render_prompt()

        # Phase 1: Show header and prompt
        draw_frame(header + "\n" + prompt)
        time.sleep(1.5)

        # Phase 2: Show initial all-masked state
        step_header = render_step_header(0, total_steps, 0, total_tokens)
        token_display = render_tokens_multiline(FINAL_TOKENS, states)

        full_frame = (
            header + "\n" + prompt + "\n"
            + step_header + "\n\n"
            + token_display + "\n"
        )
        draw_frame(full_frame)
        time.sleep(2.0)

        # Phase 3: Iterative unmasking
        for step_idx, step_data in enumerate(UNMASK_SCHEDULE, 1):
            new_indices = set()

            for token_idx, confidence in step_data:
                states[token_idx] = "revealed"
                new_indices.add(token_idx)
                revealed_count += 1

            step_header = render_step_header(
                step_idx, total_steps, revealed_count, total_tokens
            )

            # Flash animation for new tokens (3 flashes)
            for _ in range(3):
                # Flash ON
                token_display = render_tokens_multiline(
                    FINAL_TOKENS, states, new_indices
                )
                conf_panel = render_confidence_panel(step_data, FINAL_TOKENS)
                full_frame = (
                    header + "\n" + prompt + "\n"
                    + step_header + "\n\n"
                    + token_display + "\n"
                    + conf_panel + "\n"
                )
                draw_frame(full_frame)
                time.sleep(0.1)

                # Flash OFF
                token_display = render_tokens_multiline(
                    FINAL_TOKENS, states, set()
                )
                full_frame = (
                    header + "\n" + prompt + "\n"
                    + step_header + "\n\n"
                    + token_display + "\n"
                    + conf_panel + "\n"
                )
                draw_frame(full_frame)
                time.sleep(0.1)

            # Settle with highlights visible
            token_display = render_tokens_multiline(
                FINAL_TOKENS, states, new_indices
            )
            conf_panel = render_confidence_panel(step_data, FINAL_TOKENS)
            full_frame = (
                header + "\n" + prompt + "\n"
                + step_header + "\n\n"
                + token_display + "\n"
                + conf_panel + "\n"
            )
            draw_frame(full_frame)

            # Pause between steps (longer early, shorter later)
            pause = max(0.6, 1.8 - step_idx * 0.15)
            time.sleep(pause)

        # Phase 4: Final reveal
        time.sleep(0.5)
        footer = render_footer()
        final_frame = header + "\n" + prompt + "\n" + footer + "\n"
        draw_frame(final_frame)
        time.sleep(3.0)

    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write(SHOW_CURSOR)
        sys.stdout.write("\n")
        sys.stdout.flush()


# --- Entry point --------------------------------------------------------------

if __name__ == "__main__":
    if not sys.stdout.isatty():
        print("This demo requires an interactive terminal with color support.")
        print("Run directly in a terminal: python demo_script.py")
        sys.exit(1)

    try:
        cols = os.get_terminal_size().columns
        if cols < 75:
            print(f"Terminal width is {cols} columns. Recommend at least 75.")
            print("Resize your terminal for best results.")
            time.sleep(2)
    except OSError:
        pass

    print(f"{BOLD}{BR_CYAN}ChatBERT-IMR Demo{RESET}")
    print(f"{DIM}Starting in 2 seconds... (Ctrl+C to quit){RESET}")
    time.sleep(2)

    run_demo()

    print(f"\n{DIM}Demo complete. Record with: asciinema rec chatbert-demo.cast{RESET}")
