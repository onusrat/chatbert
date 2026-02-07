#!/usr/bin/env python3
"""Demo script for ChatBERT with Gradio interface."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def create_gradio_demo(model_path: str, model_type: str = "encoder_decoder"):
    """Create Gradio chat interface.

    Args:
        model_path: Path to trained model.
        model_type: Type of model.

    Returns:
        Gradio demo interface.
    """
    import gradio as gr
    from chatbert.inference.generator import ChatBERTGenerator

    print(f"Loading model from {model_path}...")
    generator = ChatBERTGenerator.from_pretrained(
        model_path,
        model_type=model_type,
        max_length=128,
        temperature=0.7,
        do_sample=True,
    )
    print("Model loaded!")

    def respond(message: str, history: list) -> str:
        """Generate response for Gradio interface."""
        # Convert Gradio history format to our format
        formatted_history = []
        for user_msg, bot_msg in history:
            formatted_history.append({"role": "user", "content": user_msg})
            if bot_msg:
                formatted_history.append({"role": "assistant", "content": bot_msg})
        formatted_history.append({"role": "user", "content": message})

        # Generate response
        response = generator.generate(formatted_history)
        return response

    # Create interface
    demo = gr.ChatInterface(
        fn=respond,
        title="ChatBERT Demo",
        description="""
        **ChatBERT**: Conversational AI with Bidirectional Encoder Representations

        This demo showcases ChatBERT, which uses BERT's bidirectional attention
        for "deliberative generation" - considering the entire response before
        committing to any part.

        Try having a conversation!
        """,
        examples=[
            "Hello! How are you today?",
            "What's the best way to learn programming?",
            "Can you tell me a joke?",
            "What do you think about artificial intelligence?",
        ],
        theme=gr.themes.Soft(),
    )

    return demo


def run_cli_chat(model_path: str, model_type: str = "encoder_decoder"):
    """Run command-line chat interface.

    Args:
        model_path: Path to trained model.
        model_type: Type of model.
    """
    from chatbert.inference.generator import ChatBERTGenerator

    print(f"Loading model from {model_path}...")
    generator = ChatBERTGenerator.from_pretrained(
        model_path,
        model_type=model_type,
        max_length=128,
        temperature=0.7,
        do_sample=True,
    )
    print("Model loaded!")

    # Start interactive chat
    generator.chat()


def run_demo_with_dummy_model():
    """Run demo with a placeholder for testing without trained model."""
    import gradio as gr

    def dummy_respond(message: str, history: list) -> str:
        """Placeholder response function."""
        return f"[Demo Mode] You said: '{message}'. This is a placeholder response. Train a model first!"

    demo = gr.ChatInterface(
        fn=dummy_respond,
        title="ChatBERT Demo (Demo Mode)",
        description="""
        **ChatBERT Demo Mode**

        No trained model found. This is a placeholder interface.

        To use with a real model:
        1. Train a model: `python scripts/train.py --config configs/ed_small.yaml`
        2. Run demo: `python scripts/demo.py --model_path ./checkpoints/chatbert-ed-small/final`
        """,
        examples=[
            "Hello!",
            "How does ChatBERT work?",
        ],
        theme=gr.themes.Soft(),
    )

    return demo


def main():
    parser = argparse.ArgumentParser(description="ChatBERT Demo")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to trained model directory",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="encoder_decoder",
        choices=["encoder_decoder", "iterative_mlm"],
        help="Type of ChatBERT model",
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Use command-line interface instead of Gradio",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public Gradio link",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for Gradio server",
    )
    parser.add_argument(
        "--demo_mode",
        action="store_true",
        help="Run in demo mode without a trained model",
    )

    args = parser.parse_args()

    if args.demo_mode:
        print("Running in demo mode (no trained model)...")
        demo = run_demo_with_dummy_model()
        demo.launch(share=args.share, server_port=args.port)
    elif args.model_path is None:
        print("No model path specified. Use --model_path or --demo_mode")
        print("Example: python scripts/demo.py --model_path ./checkpoints/chatbert-ed-small/final")
        sys.exit(1)
    elif args.cli:
        run_cli_chat(args.model_path, args.model_type)
    else:
        demo = create_gradio_demo(args.model_path, args.model_type)
        demo.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
