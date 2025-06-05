import os
import threading
from argparse import ArgumentParser
from pathlib import Path

import pyrootutils
import torch
from loguru import logger

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeTTSRequest
from tools.webui import build_app
from tools.webui.inference import get_inference_wrapper

# Make einx happy
os.environ["EINX_FILTER_TRACEBACK"] = "false"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--llama-checkpoint-path",
        type=Path,
        default="checkpoints/openaudio-s1-mini",
    )
    parser.add_argument(
        "--decoder-checkpoint-path",
        type=Path,
        default="checkpoints/openaudio-s1-mini/codec.pth",
    )
    parser.add_argument("--decoder-config-name", type=str, default="modded_dac_vq")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--max-gradio-length", type=int, default=0)
    parser.add_argument("--theme", type=str, default="dark")
    parser.add_argument("--skip-dry-run", action="store_true", default=True, help="Skip the warm-up dry run")
    parser.add_argument("--skip-model-loading", action="store_true", help="Skip model loading entirely")

    return parser.parse_args()


def load_models_in_background(args):
    """Load models in a background thread"""
    global inference_engine, app_inference_fct
    
    try:
        logger.info("Loading Llama model...")
        llama_queue = launch_thread_safe_queue(
            checkpoint_path=args.llama_checkpoint_path,
            device=args.device,
            precision=args.precision,
            compile=args.compile,
        )

        logger.info("Loading VQ-GAN model...")
        decoder_model = load_decoder_model(
            config_name=args.decoder_config_name,
            checkpoint_path=args.decoder_checkpoint_path,
            device=args.device,
        )

        logger.info("Decoder model loaded...")

        # Create the inference engine
        inference_engine = TTSInferenceEngine(
            llama_queue=llama_queue,
            decoder_model=decoder_model,
            compile=args.compile,
            precision=args.precision,
        )

        if not args.skip_dry_run:
            logger.info("Warming up...")
            # Dry run to check if the model is loaded correctly and avoid the first-time latency
            list(
                inference_engine.inference(
                    ServeTTSRequest(
                        text="Hello world.",
                        references=[],
                        reference_id=None,
                        max_new_tokens=1024,
                        chunk_length=200,
                        top_p=0.7,
                        repetition_penalty=1.5,
                        temperature=0.7,
                        format="wav",
                    )
                )
            )
            logger.info("Warming up done!")
        
        # Update the global inference function
        app_inference_fct = get_inference_wrapper(inference_engine)
        logger.info("🎉 Models loaded successfully! The web UI is now fully functional.")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        inference_engine = None
        app_inference_fct = None


if __name__ == "__main__":
    args = parse_args()
    args.precision = torch.half if args.half else torch.bfloat16

    # Check if MPS or CUDA is available
    if torch.backends.mps.is_available():
        args.device = "mps"
        logger.info("mps is available, running on mps.")
    elif not torch.cuda.is_available():
        logger.info("CUDA is not available, running on CPU.")
        args.device = "cpu"

    # Global variables to hold the inference engine and function
    inference_engine = None
    app_inference_fct = None

    def dynamic_inference_wrapper(
        text,
        reference_id,
        reference_audio,
        reference_text,
        max_new_tokens,
        chunk_length,
        top_p,
        repetition_penalty,
        temperature,
        seed,
        use_memory_cache,
    ):
        """Wrapper that checks if models are loaded before calling inference"""
        if app_inference_fct is None:
            return None, "⏳ Models are still loading, please wait..."
        return app_inference_fct(
            text,
            reference_id,
            reference_audio,
            reference_text,
            max_new_tokens,
            chunk_length,
            top_p,
            repetition_penalty,
            temperature,
            seed,
            use_memory_cache,
        )

    # Launch web UI first
    logger.info("🚀 Launching web UI...")
    app = build_app(dynamic_inference_wrapper, args.theme)
    
    # Start model loading in background if not skipped
    if not args.skip_model_loading:
        logger.info("📦 Starting model loading in background...")
        model_thread = threading.Thread(target=load_models_in_background, args=(args,))
        model_thread.daemon = True
        model_thread.start()
    else:
        logger.info("Skipping model loading as requested.")

    # Launch the app
    app.launch(show_api=True, inbrowser=True)
