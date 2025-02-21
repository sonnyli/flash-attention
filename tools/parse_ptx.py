#!/usr/bin/env python3
import sys
import re

# Regex to match the entire kernel instantiation line for "flash_fwd_kernel<...>"
# We'll capture the full template argument portion inside < ... >.
FLASH_FWD_KERNEL_RE = re.compile(
    r"(flash::flash_fwd_kernel<.*>)"
)

# Regex to extract the full argument list inside flash_fwd_kernel< ARGUMENTS >
# This will give us something like:
#  "Flash_fwd_kernel_traits<(int)128, (int)128, (int)32, (int)4, (bool)0, (bool)0, cutlass::half_t, Flash_kernel_traits<(int)128, ... >, (bool)0, (bool)0, (bool)0, (bool)0, (bool)1, (bool)1, (bool)0, (bool)0"
EXTRACT_FWD_ARGS_RE = re.compile(
    r"flash_fwd_kernel<([^>]*)>"
)

# Helper regex for parentheses blocks like (bool)0, (bool)1, (int)128, etc.
# We'll use this to do simple replacements: (bool)0 => false, (bool)1 => true, (int)N => N
PAREN_REPLACE_RE = re.compile(r"\(bool\)(0|1)|\(int\)(\d+)")

# We know the first argument is "Flash_fwd_kernel_traits< ... >",
# then 8 bools for (Is_dropout, Is_causal, Is_local, Has_alibi,
#                  Is_even_MN, Is_even_K, Is_softcap, Return_softmax).
# Then the last typename (Params) is typically not printed in your lines, 
# so we can ignore that or assume it's omitted in the PTXAS log.
#
# Inside "Flash_fwd_kernel_traits<...>" we have 8 parameters:
#   kHeadDim_, kBlockM_, kBlockN_, kNWarps_, Is_Q_in_regs_, Share_Q_K_smem_, elem_type,
#   and a further template "Flash_kernel_traits<...>" or "ArchKernelTraits<...>".
#
# We'll parse them step by step.

def bool_int_replacer(match):
    """ Map (bool)0 -> false, (bool)1 -> true, (int)123 -> 123. """
    bool_val, int_val = match.groups()
    if bool_val is not None:
        # (bool)X
        return "true" if bool_val == "1" else "false"
    else:
        # (int)N
        return int_val  # just the digits

def parse_flash_kernel_traits(arg):
    """
    Parse something like:
      Flash_kernel_traits<(int)128, (int)128, (int)32, (int)4, cutlass::half_t>
    into a named string, e.g.
      Flash_kernel_traits<kHeadDim_=128, kBlockM_=128, kBlockN_=32, kNWarps_=4, elem_type=cutlass::half_t>
    You can adapt the naming scheme to match your preference.
    """
    # Trim "Flash_kernel_traits<" and the closing ">"
    # Or if you need to be more robust, do an actual bracket parse.
    # For simplicity, let's do a quick parse:
    prefix = "Flash_kernel_traits<"
    suffix = ">"
    if arg.startswith(prefix) and arg.endswith(suffix):
        inner = arg[len(prefix) : -len(suffix)].strip()
        # The arguments inside might be separated by commas
        # We'll do a simple split at top-level commas
        # (assuming no nested templates inside these arguments)
        parts = [x.strip() for x in inner.split(",")]
        if len(parts) == 5:
            # We map them in order:
            #   0 -> kHeadDim_
            #   1 -> kBlockM_
            #   2 -> kBlockN_
            #   3 -> kNWarps_
            #   4 -> elem_type
            return (
                "Flash_kernel_traits<"
                f"kHeadDim_={parts[0]}, "
                f"kBlockM_={parts[1]}, "
                f"kBlockN_={parts[2]}, "
                f"kNWarps_={parts[3]}, "
                f"elem_type={parts[4]}"
                ">"
            )
    # If for some reason it doesn't match, just return as-is.
    return arg

def parse_fwd_kernel_traits(arg):
    """
    Parse something like:
      Flash_fwd_kernel_traits<(int)128, (int)128, (int)32, (int)4, (bool)0, (bool)0, cutlass::half_t, Flash_kernel_traits<(int)128, ...>>
    Return a string with named parameters.
    """
    prefix = "Flash_fwd_kernel_traits<"
    suffix = ">"
    if arg.startswith(prefix) and arg.endswith(suffix):
        inner = arg[len(prefix) : -len(suffix)].strip()
        # Again, a simple top-level split by commas:
        parts = [x.strip() for x in split_top_level_commas(inner)]
        # Expecting 8 arguments. The last one is itself a template, e.g. Flash_kernel_traits<...>
        # According to your signature:
        #   1) kHeadDim_
        #   2) kBlockM_
        #   3) kBlockN_
        #   4) kNWarps_
        #   5) Is_Q_in_regs_
        #   6) Share_Q_K_smem_
        #   7) elem_type
        #   8) The next template type (e.g. Flash_kernel_traits<...>)
        if len(parts) == 8:
            # Last part might be "Flash_kernel_traits<...>"
            arch_traits = parse_flash_kernel_traits(parts[7])
            return (
                "Flash_fwd_kernel_traits<"
                f"kHeadDim_={parts[0]}, "
                f"kBlockM_={parts[1]}, "
                f"kBlockN_={parts[2]}, "
                f"kNWarps_={parts[3]}, "
                f"Is_Q_in_regs_={parts[4]}, "
                f"Share_Q_K_smem_={parts[5]}, "
                f"elem_type={parts[6]}, "
                f"{arch_traits}"
                ">"
            )
    # Default fallback if something doesn't match.
    return arg

def split_top_level_commas(s):
    """
    Split on commas that are not nested in < >.
    A small utility to handle nested templates properly.
    """
    parts = []
    bracket_level = 0
    start = 0
    for i, ch in enumerate(s):
        if ch == '<':
            bracket_level += 1
        elif ch == '>':
            bracket_level -= 1
        elif ch == ',' and bracket_level == 0:
            parts.append(s[start:i])
            start = i + 1
    parts.append(s[start:])
    return parts

def parse_and_rewrite(line):
    """
    1) Find 'flash::flash_fwd_kernel< ... >'
    2) Extract arguments
    3) Rewrite with parameter names
    """
    match = FLASH_FWD_KERNEL_RE.search(line)
    if not match:
        return line  # No change if the pattern isn't found

    # The portion "flash_fwd_kernel<...>"
    kernel_text = match.group(1)

    # Extract everything in the angle brackets after flash_fwd_kernel
    m2 = EXTRACT_FWD_ARGS_RE.search(kernel_text)
    if not m2:
        return line  # fallback

    arg_text = m2.group(1).strip()

    # Now we have something like:
    #   "Flash_fwd_kernel_traits<(int)128, (int)128, (int)32, (int)4, (bool)0, (bool)0, cutlass::half_t, Flash_kernel_traits<(int)128, ...>>, (bool)0, (bool)0, (bool)0, (bool)0, (bool)1, (bool)1, (bool)0, (bool)0"

    # Split out the first big trait from the trailing bools
    # Because the first chunk is itself a template of the form Flash_fwd_kernel_traits<...>
    # We'll do a top-level split on commas, but only after we isolate that entire trait.
    # Let's do that with split_top_level_commas again:
    parts = split_top_level_commas(arg_text)

    # The very first argument is "Flash_fwd_kernel_traits<...>", the rest should be the 8 bools
    if len(parts) < 9:
        # If something is off or we have fewer arguments, just bail out
        return line

    kernel_traits_str = parts[0].strip()
    bool_params = parts[1:]  # the 8 booleans

    # parse the kernel_traits_str
    new_kernel_traits = parse_fwd_kernel_traits(kernel_traits_str)

    # Now map (bool)0->false, (bool)1->true, etc. in the boolean parameters
    # We have 8 booleans in order:
    #   1) Is_dropout
    #   2) Is_causal
    #   3) Is_local
    #   4) Has_alibi
    #   5) Is_even_MN
    #   6) Is_even_K
    #   7) Is_softcap
    #   8) Return_softmax
    bool_labels = [
        "Is_dropout",
        "Is_causal",
        "Is_local",
        "Has_alibi",
        "Is_even_MN",
        "Is_even_K",
        "Is_softcap",
        "Return_softmax"
    ]
    if len(bool_params) != len(bool_labels):
        return line

    bool_rewritten = []
    for label, bp in zip(bool_labels, bool_params):
        bp_clean = PAREN_REPLACE_RE.sub(bool_int_replacer, bp.strip())
        bool_rewritten.append(f"{label}={bp_clean}")

    # Put it all together into a new string
    new_text = (
        f"flash::flash_fwd_kernel<\n"
        f"  Kernel_traits = {new_kernel_traits},\n"
        f"  {', '.join(bool_rewritten)}\n"
        f">"
    )

    # Replace the old substring in the line with new_text
    new_line = line.replace(kernel_text, new_text)
    return new_line

def main():
    # If the script is called with a file argument, read from it;
    # otherwise, read from stdin.
    if len(sys.argv) > 1:
        filenames = sys.argv[1:]
    else:
        filenames = [None]  # None => stdin

    for fname in filenames:
        if fname is None:
            # reading from stdin
            for line in sys.stdin:
                line = line.rstrip("\n")
                print(parse_and_rewrite(line))
        else:
            with open(fname, 'r') as f:
                for line in f:
                    line = line.rstrip("\n")
                    print(parse_and_rewrite(line))


if __name__ == "__main__":
    main()