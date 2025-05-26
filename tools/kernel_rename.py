#!/usr/bin/env python3
import re


def extract_kernel_info(name):
    """Extract the key information from the kernel name"""
    formats = {}

    # Use a simpler approach - just extract the key numbers
    head_dim = extract_number(name, r"kHeadDim_=(\d+)")
    if not head_dim:
        head_dim = extract_number(name, r"\(int\)(\d+)")

    block_m = extract_number(name, r"kBlockM_=(\d+)")
    if not block_m:
        # Try to get second number
        matches = re.findall(r"\(int\)(\d+)", name)
        if len(matches) >= 2:
            block_m = matches[1]

    block_n = extract_number(name, r"kBlockN_=(\d+)")
    if not block_n:
        # Try to get third number
        matches = re.findall(r"\(int\)(\d+)", name)
        if len(matches) >= 3:
            block_n = matches[2]

    # Element type
    if "half_t" in name:
        elem_type = "fp16"
    elif "bfloat16_t" in name:
        elem_type = "bf16"
    else:
        elem_type = None

    # Check if causal
    causal = "Is_causal=true" in name or (", (bool)0, (bool)1," in name)

    # Check if even_mn
    even_mn = "Is_even_MN=true" in name or (", (bool)1, (bool)1," in name)

    return {
        "head_dim": head_dim,
        "block_m": block_m,
        "block_n": block_n,
        "elem_type": elem_type,
        "causal": causal,
        "even_mn": even_mn,
    }


def extract_number(text, pattern):
    """Extract a number using a regex pattern"""
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return None


def format_kernel_name(name):
    """Format the kernel name to be more readable"""
    # Don't process if not a flash kernel
    if "flash::flash_fwd_kernel<" not in name:
        return name

    # Extract key information
    info = extract_kernel_info(name)

    # If we have all the essential information, format it nicely
    if (
        info["head_dim"]
        and info["block_m"]
        and info["block_n"]
        and info["elem_type"]
    ):
        result = f"flash_kernel[{info['head_dim']}_{info['block_m']}x{info['block_n']}_{info['elem_type']}"

        # Add flags
        flags = []
        if info["causal"]:
            flags.append("causal")
        if info["even_mn"]:
            flags.append("even_mn")

        if flags:
            result += "_" + "_".join(flags)

        result += "]"
        return result

    # Fallback - just return a shorter version of the original name
    if len(name) > 60:
        return name[:57] + "..."
    return name


# Test the function
if __name__ == "__main__":
    test_name = "flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (int)4, (bool)0, (bool)0, cutlass::half_t, Flash_kernel_traits<(int)128, (int)128, (int)64, (int)4, cutlass::half_t>>, (bool)0, (bool)1, (bool)0, (bool)0, (bool)1, (bool)1, (bool)0, (bool)0>"

    formatted = format_kernel_name(test_name)
    print(f"Original: {test_name}")
    print(f"Formatted: {formatted}")
