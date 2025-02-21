#!/usr/bin/env python3
import re
import sys
from dataclasses import dataclass


@dataclass
class KernelFunction:
    name: str
    architecture: str = ""
    used_registers: int = 0
    used_barriers: int = 0
    cmem_bytes: int = 0
    compile_time_ms: float = 0.0
    spill_stores_bytes: int = 0
    spill_loads_bytes: int = 0
    stack_frame_bytes: int = 0
    # Template parameters
    head_dim: int = 0
    block_m: int = 0
    block_n: int = 0
    n_warps: int = 0
    q_in_regs: bool = False
    share_qk_smem: bool = False
    element_type: str = ""
    is_dropout: bool = False
    is_causal: bool = False
    is_local: bool = False
    has_alibi: bool = False
    is_even_mn: bool = False
    is_even_k: bool = False
    is_softcap: bool = False
    return_softmax: bool = False

    def has_spills(self):
        return self.spill_stores_bytes > 0 or self.spill_loads_bytes > 0

    def __str__(self):
        """Format the kernel in the flash(param=value, ...) format"""
        params = []
        if self.head_dim:
            params.append(f"head_dim={self.head_dim}")
        if self.block_m:
            params.append(f"block_m={self.block_m}")
        if self.block_n:
            params.append(f"block_n={self.block_n}")
        if self.n_warps:
            params.append(f"warps={self.n_warps}")
        if self.element_type:
            params.append(f"type={self.element_type}")

        # Boolean parameters (only include if true)
        if self.q_in_regs:
            params.append("q_in_regs=true")
        if self.share_qk_smem:
            params.append("share_qk_smem=true")
        if self.is_dropout:
            params.append("dropout=true")
        if self.is_causal:
            params.append("causal=true")
        if self.is_local:
            params.append("local=true")
        if self.has_alibi:
            params.append("alibi=true")
        if self.is_even_mn:
            params.append("even_mn=true")
        if self.is_even_k:
            params.append("even_k=true")
        if self.is_softcap:
            params.append("softcap=true")
        if self.return_softmax:
            params.append("return_softmax=true")

        return f"flash({', '.join(params)})"


def bool_int_replacer(match):
    """Replace (bool)0 -> false, (bool)1 -> true, (int)X -> X"""
    text = match.group(0)
    if text.startswith("(bool)"):
        val = text[len("(bool)") :]
        return "true" if val == "1" else "false"
    elif text.startswith("(int)"):
        val = text[len("(int)") :]
        return val
    return text


def balanced_split(text, sep=",", open_chars="<(", close_chars=">)"):
    """Split text by separator, respecting nested parentheses and brackets"""
    parts = []
    current = []
    stack = []

    for char in text:
        if not stack and char == sep:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
            if char in open_chars:
                stack.append(char)
            elif char in close_chars and stack:
                open_char = stack.pop()
                if (open_char == "<" and char != ">") or (
                    open_char == "(" and char != ")"
                ):
                    stack.append(open_char)  # not a matching close, push back

    if current:
        parts.append("".join(current).strip())

    return parts


def extract_kernel_args(text):
    """Extract template arguments from flash_fwd_kernel<...>"""
    # Find the start and end of the template
    start = text.find("flash::flash_fwd_kernel<") + len(
        "flash::flash_fwd_kernel<"
    )

    # Find the matching closing '>'
    stack = []
    end = -1
    for i in range(start, len(text)):
        if text[i] == "<":
            stack.append("<")
        elif text[i] == ">":
            if not stack:
                end = i
                break
            stack.pop()

    if end == -1:
        return None  # Couldn't find the matching '>'

    return text[start:end]


def parse_flash_kernel_traits(trait_text):
    """Parse Flash_kernel_traits<...> into a formatted string"""
    if not trait_text.startswith(
        "Flash_kernel_traits<"
    ) or not trait_text.endswith(">"):
        return trait_text

    # Extract inner parts
    inner = trait_text[len("Flash_kernel_traits<") : -1]
    parts = balanced_split(inner)

    if len(parts) != 5:
        return trait_text

    # Apply replacements to each part
    parts = [
        re.sub(r"\(bool\)\d|\(int\)\d+", bool_int_replacer, p) for p in parts
    ]

    return (
        f"Flash_kernel_traits<"
        f"kHeadDim_={parts[0]}, "
        f"kBlockM_={parts[1]}, "
        f"kBlockN_={parts[2]}, "
        f"kNWarps_={parts[3]}, "
        f"elem_type={parts[4]}"
        f">"
    )


def parse_fwd_kernel_traits(trait_text):
    """Parse Flash_fwd_kernel_traits<...> into a formatted string"""
    if not trait_text.startswith(
        "Flash_fwd_kernel_traits<"
    ) or not trait_text.endswith(">"):
        return trait_text

    # Extract inner parts
    inner = trait_text[len("Flash_fwd_kernel_traits<") : -1]
    parts = balanced_split(inner)

    if len(parts) != 8:
        return trait_text

    # Apply replacements to the numeric parts
    for i in range(6):
        parts[i] = re.sub(r"\(bool\)\d|\(int\)\d+", bool_int_replacer, parts[i])

    # Parse the nested kernel traits (last element)
    parts[7] = parse_flash_kernel_traits(parts[7])

    return (
        f"Flash_fwd_kernel_traits<"
        f"kHeadDim_={parts[0]}, "
        f"kBlockM_={parts[1]}, "
        f"kBlockN_={parts[2]}, "
        f"kNWarps_={parts[3]}, "
        f"Is_Q_in_regs_={parts[4]}, "
        f"Share_Q_K_smem_={parts[5]}, "
        f"elem_type={parts[6]}, "
        f"{parts[7]}"
        f">"
    )


def parse_line(line):
    """Parse a line containing flash_fwd_kernel<...> and reformat it"""
    # For non-kernel/non-stats lines, return None to filter them out
    if not is_related_line(line):
        return None

    # If it's related but not a kernel line itself, return as is
    if "flash::flash_fwd_kernel<" not in line:
        return line

    # Extract the kernel args
    kernel_text_start = line.find("flash::flash_fwd_kernel<")
    kernel_text_end = line.rfind("'")
    if kernel_text_end == -1:
        kernel_text_end = len(line)

    kernel_text = line[kernel_text_start:kernel_text_end]
    kernel_args = extract_kernel_args(kernel_text)

    if not kernel_args:
        return line  # Return original line if we couldn't parse it

    # Split into Flash_fwd_kernel_traits<...> and boolean flags
    parts = balanced_split(kernel_args)

    if len(parts) != 9:  # First part is kernel traits, then 8 booleans
        return line  # Return original line if it doesn't match expected format

    # Parse the kernel traits
    kernel_traits = parse_fwd_kernel_traits(parts[0])

    # Parse the boolean flags
    bool_labels = [
        "Is_dropout",
        "Is_causal",
        "Is_local",
        "Has_alibi",
        "Is_even_MN",
        "Is_even_K",
        "Is_softcap",
        "Return_softmax",
    ]

    bool_values = []
    for i, (label, value) in enumerate(zip(bool_labels, parts[1:])):
        value = re.sub(r"\(bool\)\d", bool_int_replacer, value)
        bool_values.append(f"{label}={value}")

    # Create the new formatted kernel text
    new_kernel_text = (
        f"flash::flash_fwd_kernel<\n"
        f"  Kernel_traits = {kernel_traits},\n"
        f"  {', '.join(bool_values)}\n"
        f">"
    )

    # Replace in the original line
    return line[:kernel_text_start] + new_kernel_text + line[kernel_text_end:]


def is_related_line(line):
    """Check if a line is related to flash kernel or contains important information"""
    if "flash::flash_fwd_kernel<" in line:
        return True

    # Lines with function properties, but only if it's for a flash kernel
    if "Function properties for flash::flash_fwd_kernel<" in line:
        return True

    # Include register usage and other stats lines that follow a kernel line
    if any(
        x in line
        for x in [
            "bytes stack frame",
            "Used",
            "registers",
            "barriers",
            "Compile time",
        ]
    ):
        return True

    return False


def format_kernel_name(kernel_name):
    """Extract template parameters from the kernel name and return a dictionary of parameters"""
    params = {
        "head_dim": 0,
        "block_m": 0,
        "block_n": 0,
        "n_warps": 0,
        "q_in_regs": False,
        "share_qk_smem": False,
        "element_type": "",
        "is_dropout": False,
        "is_causal": False,
        "is_local": False,
        "has_alibi": False,
        "is_even_mn": False,
        "is_even_k": False,
        "is_softcap": False,
        "return_softmax": False,
    }

    # Extract the Flash_fwd_kernel_traits part
    traits_match = re.search(r"Flash_fwd_kernel_traits<([^>]*)>", kernel_name)
    if not traits_match:
        return params

    traits_content = traits_match.group(1)

    # Split the traits content by commas, respecting nested <> brackets
    parts = balanced_split(traits_content)
    if len(parts) < 8:
        return params

    # Extract numeric parameters
    try:
        # Extract the values from (int)X format
        int_pattern = r"\(int\)(\d+)"
        params["head_dim"] = int(re.search(int_pattern, parts[0]).group(1))
        params["block_m"] = int(re.search(int_pattern, parts[1]).group(1))
        params["block_n"] = int(re.search(int_pattern, parts[2]).group(1))
        params["n_warps"] = int(re.search(int_pattern, parts[3]).group(1))
    except (AttributeError, IndexError):
        pass

    # Extract boolean parameters inside the traits
    params["q_in_regs"] = "(bool)1" in parts[4]
    params["share_qk_smem"] = "(bool)1" in parts[5]

    # Element type
    if "half_t" in parts[6]:
        params["element_type"] = "fp16"
    elif "bfloat16_t" in parts[6]:
        params["element_type"] = "bf16"

    # Extract the boolean flags outside the traits
    flags_match = re.search(
        r"Flash_fwd_kernel_traits<[^>]*>>,\s*(.+?)(?='|$)", kernel_name
    )
    if flags_match:
        flags_str = flags_match.group(1)
        flags = balanced_split(flags_str)

        if len(flags) >= 8:
            params["is_dropout"] = "(bool)1" in flags[0]
            params["is_causal"] = "(bool)1" in flags[1]
            params["is_local"] = "(bool)1" in flags[2]
            params["has_alibi"] = "(bool)1" in flags[3]
            params["is_even_mn"] = "(bool)1" in flags[4]
            params["is_even_k"] = "(bool)1" in flags[5]
            params["is_softcap"] = "(bool)1" in flags[6]
            params["return_softmax"] = "(bool)1" in flags[7]

    return params


def format_kernel_function(kernel, index):
    """Format kernel function object into the desired output format"""
    # Use the string representation directly
    readable_name = str(kernel)

    output = [f"=== Function #{index} ==="]
    output.append(f"  Name        : {readable_name}")
    output.append(f"  Architecture          : {kernel.architecture}")
    output.append(f"  Used registers        : {kernel.used_registers}")
    output.append(f"  Used barriers         : {kernel.used_barriers}")
    output.append(f"  cmem[0] (bytes)       : {kernel.cmem_bytes}")
    output.append(f"  Compile time (ms)     : {kernel.compile_time_ms}")

    # Only include spill info if there are spills
    if kernel.has_spills():
        output.append(f"  Spill stores (bytes)  : {kernel.spill_stores_bytes}")
        output.append(f"  Spill loads (bytes)   : {kernel.spill_loads_bytes}")

    return "\n".join(output)


def extract_kernel_name(line):
    """Extract kernel name from compiling or function properties line"""
    match = re.search(r"'(flash::flash_fwd_kernel<[^']*)'", line)
    if match:
        return match.group(1)
    return ""


def extract_arch(line):
    """Extract architecture from compiling line"""
    match = re.search(r"for '(sm_\d+)'", line)
    if match:
        return match.group(1)
    return ""


def extract_number(line, pattern):
    """Extract a number from a line matching a pattern"""
    match = re.search(pattern, line)
    if match:
        return match.group(1)
    return "0"


def parse_kernel_info(lines):
    """Parse kernel information from a group of lines"""
    kernel_functions = {}
    current_kernel = None

    for line in lines:
        # New kernel definition
        if "Compiling entry function" in line:
            kernel_name = extract_kernel_name(line)
            arch = extract_arch(line)

            if kernel_name:
                current_kernel = kernel_name

                # Create new kernel and extract template parameters
                kernel = KernelFunction(name=kernel_name, architecture=arch)
                template_params = format_kernel_name(kernel_name)

                # Update kernel with template parameters
                for param, value in template_params.items():
                    setattr(kernel, param, value)

                kernel_functions[current_kernel] = kernel

        # Function properties
        elif "Function properties for" in line:
            kernel_name = extract_kernel_name(line)
            if kernel_name:
                current_kernel = kernel_name
                if kernel_name not in kernel_functions:
                    kernel = KernelFunction(name=kernel_name)

                    # Extract template parameters
                    template_params = format_kernel_name(kernel_name)

                    # Update kernel with template parameters
                    for param, value in template_params.items():
                        setattr(kernel, param, value)

                    kernel_functions[current_kernel] = kernel

        # Stack frame
        elif "bytes stack frame" in line:
            if current_kernel and current_kernel in kernel_functions:
                parts = line.strip().split(",")
                for part in parts:
                    if "stack frame" in part:
                        kernel_functions[
                            current_kernel
                        ].stack_frame_bytes = int(
                            extract_number(part, r"(\d+) bytes stack frame")
                        )
                    elif "spill stores" in part:
                        kernel_functions[
                            current_kernel
                        ].spill_stores_bytes = int(
                            extract_number(part, r"(\d+) bytes spill stores")
                        )
                    elif "spill loads" in part:
                        kernel_functions[
                            current_kernel
                        ].spill_loads_bytes = int(
                            extract_number(part, r"(\d+) bytes spill loads")
                        )

        # Register usage
        elif "Used" in line and "registers" in line:
            if current_kernel and current_kernel in kernel_functions:
                parts = line.strip().split(",")
                for part in parts:
                    if "registers" in part:
                        kernel_functions[current_kernel].used_registers = int(
                            extract_number(part, r"Used (\d+) registers")
                        )
                    elif "barriers" in part:
                        kernel_functions[current_kernel].used_barriers = int(
                            extract_number(part, r"used (\d+) barriers")
                        )
                    elif "cmem" in part:
                        kernel_functions[current_kernel].cmem_bytes = int(
                            extract_number(part, r"(\d+) bytes cmem")
                        )

        # Compile time
        elif "Compile time" in line:
            if current_kernel and current_kernel in kernel_functions:
                kernel_functions[current_kernel].compile_time_ms = float(
                    extract_number(line, r"Compile time = (\d+\.\d+)")
                )

    return kernel_functions


def main():
    # If the script is called with a file argument, read from it;
    # otherwise, read from stdin.
    if len(sys.argv) > 1:
        filenames = sys.argv[1:]
    else:
        filenames = [None]  # None => stdin

    for fname in filenames:
        if fname is None:
            # Reading from stdin
            lines = [line.rstrip("\n") for line in sys.stdin]
        else:
            # Reading from file
            with open(fname, "r") as f:
                lines = [line.rstrip("\n") for line in f]

        # Filter to only related lines
        related_lines = []
        for line in lines:
            if is_related_line(line):
                related_lines.append(line)

        # Parse kernel info from related lines
        kernel_functions = parse_kernel_info(related_lines)

        # Print kernel info in the desired format
        for i, (_, kernel) in enumerate(kernel_functions.items(), 1):
            print(format_kernel_function(kernel, i))
            print()  # Empty line between functions


if __name__ == "__main__":
    main()
