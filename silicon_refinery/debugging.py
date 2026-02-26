import asyncio
import concurrent.futures
import functools
import importlib
import inspect
import logging
import sys
import traceback
from pathlib import Path

from .exceptions import AppleFMSetupError, ensure_model_available
from .protocols import create_model, create_session

fm = importlib.import_module("apple_fm_sdk")

logger = logging.getLogger("silicon_refinery.debug")
_SYNC_ANALYSIS_TIMEOUT_SECONDS = 30


@fm.generable()
class DebuggingAnalysis:
    error_summary: str = fm.guide(description="A brief summary of what went wrong.")
    possible_causes: list[str] = fm.guide(
        description="List of likely root causes based on the traceback."
    )
    certainty_level: str = fm.guide(
        anyOf=["LOW", "MEDIUM", "HIGH"], description="Confidence in the diagnosis."
    )
    suggested_fix: str = fm.guide(description="Actionable steps to fix the issue.")


def _run_analysis_sync_once(
    exc: Exception,
    func_name: str,
    summary_to: str | None,
    prompt_to: str | None,
    silenced: bool,
    summary_log_level: str | int,
    prompt_log_level: str | int,
) -> None:
    coro = _handle_exception(
        exc,
        func_name,
        summary_to,
        prompt_to,
        silenced,
        summary_log_level,
        prompt_log_level,
    )
    try:
        asyncio.run(coro)
    finally:
        coro.close()


def _run_analysis_sync_best_effort(
    exc: Exception,
    func_name: str,
    summary_to: str | None,
    prompt_to: str | None,
    silenced: bool,
    summary_log_level: str | int,
    prompt_log_level: str | int,
) -> None:
    try:
        _run_analysis_sync_once(
            exc,
            func_name,
            summary_to,
            prompt_to,
            silenced,
            summary_log_level,
            prompt_log_level,
        )
        return
    except RuntimeError:
        # asyncio.run() can fail if the current thread already has a running loop.
        pass
    except Exception:
        logger.warning("[SiliconRefinery Debug] AI analysis failed.", exc_info=True)
        return

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(
            _run_analysis_sync_once,
            exc,
            func_name,
            summary_to,
            prompt_to,
            silenced,
            summary_log_level,
            prompt_log_level,
        )
        try:
            future.result(timeout=_SYNC_ANALYSIS_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            future.cancel()
            logger.warning("[SiliconRefinery Debug] AI analysis timed out.")
        except Exception:
            logger.warning("[SiliconRefinery Debug] AI analysis failed.", exc_info=True)
    finally:
        # Do not block original exception propagation on background analysis shutdown.
        executor.shutdown(wait=False, cancel_futures=True)


def enhanced_debug(
    summary_to: str | None = "stderr",
    prompt_to: str | None = "stdout",
    silenced: bool = False,
    summary_log_level: str | int = "error",
    prompt_log_level: str | int = "info",
):
    """
    A decorator that catches exceptions, prints the traceback, and invokes the Apple Foundation Model
    to perform a detailed root-cause analysis.

    Args:
        summary_to: Where to output the analysis ("stdout", "stderr", "log", or None to silence).
        prompt_to: Where to route the detailed prompt payload. Use "stdout" (default), a
                   relative/absolute file path (written as .txt), "log", or None to silence prompt output.
        silenced: If True, silences all enhanced_debug outputs and skips analysis.
        summary_log_level: Logging level when summary_to="log" (e.g. "info", "warning", "error").
        prompt_log_level: Logging level when prompt_to="log" (e.g. "debug", "info", "warning").
    """

    def decorator(func):
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    try:
                        await _handle_exception(
                            e,
                            func.__name__,
                            summary_to,
                            prompt_to,
                            silenced,
                            summary_log_level,
                            prompt_log_level,
                        )
                    except Exception:
                        logger.warning("[SiliconRefinery Debug] AI analysis failed.", exc_info=True)
                    raise

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    _run_analysis_sync_best_effort(
                        e,
                        func.__name__,
                        summary_to,
                        prompt_to,
                        silenced,
                        summary_log_level,
                        prompt_log_level,
                    )
                    raise

            return sync_wrapper

    return decorator


def _resolve_log_level(level: str | int, fallback: int = logging.ERROR) -> int:
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        if level.isdigit():
            return int(level)
        resolved = logging.getLevelName(level.upper())
        if isinstance(resolved, int):
            return resolved
    logger.warning(
        "[SiliconRefinery Debug] Unsupported log level '%s'; falling back to %s.",
        level,
        logging.getLevelName(fallback),
    )
    return fallback


def _log_text(message: str, level: str | int, fallback: int = logging.ERROR) -> None:
    logger.log(_resolve_log_level(level, fallback), message)


def _write_prompt_output(prompt_content: str, prompt_to: str, prompt_log_level: str | int) -> None:
    if prompt_to == "stdout":
        print(prompt_content)
        return
    if prompt_to == "stderr":
        print(prompt_content, file=sys.stderr)
        return
    if prompt_to == "log":
        _log_text(prompt_content, prompt_log_level, fallback=logging.INFO)
        return

    try:
        prompt_path = Path(prompt_to).expanduser()
        if prompt_path.name in {"", ".", ".."}:
            raise ValueError("prompt_to must point to a writable file path")
        prompt_path = prompt_path.with_suffix(".txt")
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(prompt_content, encoding="utf-8")
        print(f"Generated AI Agent Prompt written to: {prompt_path}\n")
    except (OSError, TypeError, ValueError):
        logger.warning(
            "[SiliconRefinery Debug] Could not write prompt_to path '%s'.",
            prompt_to,
            exc_info=True,
        )


async def _handle_exception(
    exc: Exception,
    func_name: str,
    summary_to: str | None = "stderr",
    prompt_to: str | None = "stdout",
    silenced: bool = False,
    summary_log_level: str | int = "error",
    prompt_log_level: str | int = "info",
):
    """Core logic to analyze the exception with the local FM."""
    if silenced:
        return

    tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

    # Print original traceback
    print(f"\n--- Exception caught in '{func_name}' ---", file=sys.stderr)
    print(tb_str, file=sys.stderr)
    print("--- End of standard traceback ---\n", file=sys.stderr)

    model = create_model()
    try:
        ensure_model_available(model, context="enhanced_debug")
    except AppleFMSetupError as setup_error:
        logger.error("%s", setup_error)
        return

    session = create_session(
        instructions="You are an expert Principal Software Engineer. Analyze the following Python traceback. Identify the root cause, provide a certainty level, and suggest a fix.",
        model=model,
    )

    print("SiliconRefinery is analyzing the crash locally via Neural Engine...", file=sys.stderr)

    try:
        analysis = await session.respond(tb_str, generating=DebuggingAnalysis)

        output = [
            "\n" + "=" * 50,
            f"SiliconRefinery AI Debug Analysis (Certainty: {analysis.certainty_level})",
            "=" * 50,
            f"Summary: {analysis.error_summary}\n",
            "Possible Causes:",
        ]
        for idx, cause in enumerate(analysis.possible_causes, 1):
            output.append(f"  {idx}. {cause}")
        output.append(f"\nSuggested Fix: {analysis.suggested_fix}")
        output.append("=" * 50 + "\n")

        output_str = "\n".join(output)

        if summary_to == "stdout":
            print(output_str)
        elif summary_to == "stderr":
            print(output_str, file=sys.stderr)
        elif summary_to == "log":
            _log_text(output_str, summary_log_level, fallback=logging.ERROR)
        elif summary_to is not None:
            logger.warning(
                "[SiliconRefinery Debug] Unsupported summary_to value '%s'; expected 'stdout', 'stderr', 'log', or None.",
                summary_to,
            )

        if prompt_to is not None:
            prompt_content = f"""I encountered a crash in my Python application. Here is the traceback:
```python
{tb_str}
```

A local AI agent performed a preliminary analysis and concluded:
- Summary: {analysis.error_summary}
- Possible Causes: {", ".join(analysis.possible_causes)}
- Suggested Fix: {analysis.suggested_fix}

Please act as an expert developer (like Jeff Dean or a top-tier engineer). Provide a comprehensive, patched version of the code that resolves this issue, along with any necessary explanations.
"""
            _write_prompt_output(prompt_content, prompt_to, prompt_log_level)

    except Exception as e:
        print(f"SiliconRefinery AI analysis failed: {e}", file=sys.stderr)
