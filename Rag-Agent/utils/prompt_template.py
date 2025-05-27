from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re

def get_prompt():
    return ChatPromptTemplate.from_template("""
You are a helpful assistant specialized in Docker CLI commands.

Context:
{context}

User Query: {input}

Please respond ONLY with the exact Docker CLI commands relevant to the question. Do NOT include any explanations or extra text.
Separate multiple commands by new lines.
                                            
OUTPUT INSTRUCTIONS:
- If enough information is provided, return ONLY the valid CLI command.
- If not, ask the user *exactly what is missing* to generate the command.
""")

import re

class DockerCommandParser(StrOutputParser):
    def parse(self, text: str) -> str:
        # If the response does not contain any newline characters, return it as-is
        if '\n' not in text:
            return text.strip()

        # Extract content between triple backticks
        code_blocks = re.findall(r"```(?:\w*\n)?(.*?)```", text, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()

        # Fallback: extract lines that start with 'docker'
        lines = text.strip().splitlines()
        command_lines = [line.strip() for line in lines if line.strip().startswith("docker")]
        return "\n".join(command_lines)


def get_parser():
    return DockerCommandParser()