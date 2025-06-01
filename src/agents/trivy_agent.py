import subprocess
import os
import tempfile

from agents.generation_agent import GenerationModes
from llm_provider import SupportedLLMs

class TrivyAgent:

    def generate_sbom(self, directory_path, output_path):
        result = subprocess.run(
            [os.getenv("TRIVY_PATH"), "fs", "--format", "cyclonedx", "--output", output_path, directory_path],
            capture_output=True,
            text=True,
            encoding="utf-8"
        )
        return result.returncode == 0

    def scan_with_trivy(self, directory_path):
        result = subprocess.run(
            [os.getenv("TRIVY_PATH"), "fs", directory_path],
            capture_output=True,
            text=True,
            encoding="utf-8"
        )
        return result.stdout

    def invoke(self, state):
        trivy_results = {}
        for llm_codeblock_dict in state["llm_codeblocks_dicts"]:
            for llm, codeblocks in llm_codeblock_dict.items():
                if llm not in SupportedLLMs:
                    raise ValueError(f"Unexpected LLM: {llm}")

                with tempfile.TemporaryDirectory() as tmpdir:
                    for tech_block in codeblocks:
                        blocks = tech_block.get('blocks', [])
                        for cb in blocks:
                            filename = os.path.basename(cb.get('filename'))
                            code = cb.get('code')
                            if not filename or not code:
                                continue
                            target_file = os.path.join(tmpdir, filename)
                            with open(target_file, "w", encoding="utf-8") as f:
                                f.write(code)

                    result = self.scan_with_trivy(tmpdir)
                    trivy_results[llm] = result
        return {
            "trivy_results": trivy_results
        }
