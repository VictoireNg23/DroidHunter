"""Smali parsing: one Dalvik instruction -> (opcode, op1, op2).

op1 groups every register of the instruction ("v0|v2"), as in §3.2 of the paper
where op1 can be two-dimensional. op2 is the first non-register operand
(method, field, type...). Literals that carry no structure are reduced to
placeholders so that strings or constants do not blow up the vocabulary.
"""
import re
from pathlib import Path

NONE = "<NONE>"
STR = "<STR>"
LIT = "<LIT>"
LABEL = "<LABEL>"

REGISTER = re.compile(r"^[vp]\d+$")
SMALI_DIR = re.compile(r"^smali(?:_classes(\d+))?$")


def load_opcodes(path):
    """ops.txt has trailing spaces and duplicated entries; keep the unique set."""
    return frozenset(x.strip() for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip())


def split_operands(text):
    """Split on top-level commas, ignoring commas inside {...} and "..."."""
    parts, buf, depth, quoted, escaped = [], [], 0, False, False
    for ch in text:
        if quoted:
            buf.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                quoted = False
            continue
        if ch == '"':
            quoted = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    return [p for p in parts if p]


def _registers(token):
    if token.startswith("{"):
        inner = token.strip("{}").strip()
        if not inner:
            return []
        if ".." in inner:  # register range {v0 .. v5}: keep both ends
            return [r.strip() for r in inner.split("..")]
        return [r.strip() for r in inner.split(",")]
    return [token] if REGISTER.match(token) else None


def _normalise_operand(token):
    if token.startswith('"'):
        return STR
    if token.startswith(":"):
        return LABEL
    if token[0] in "-0123456789" or token.startswith(("#", "0x")):
        return LIT
    return token


def parse_instruction(mnemonic, operands):
    regs, second = [], None
    for token in split_operands(operands):
        found = _registers(token)
        if found is not None:
            regs.extend(found)
        elif second is None:
            second = _normalise_operand(token)
    return mnemonic, "|".join(regs) if regs else NONE, second or NONE


def iter_smali_instructions(decoded_dir, opcodes):
    """Yield (opcode, op1, op2) for every instruction inside .method blocks.

    Dex folders are read in multidex order (smali, smali_classes2, ...) and files in
    sorted order so that the sequence is deterministic across runs and machines.
    """
    decoded_dir = Path(decoded_dir)
    dex_dirs = []
    for p in decoded_dir.iterdir():
        m = SMALI_DIR.match(p.name)
        if p.is_dir() and m:
            dex_dirs.append((int(m.group(1) or 1), p))
    if not dex_dirs:
        raise ValueError("apktool produced no smali folder")
    for _, dex in sorted(dex_dirs):
        for path in sorted(dex.rglob("*.smali")):
            in_method = False
            with path.open(encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith(".method"):
                        in_method = True
                        continue
                    if line.startswith(".end method"):
                        in_method = False
                        continue
                    if not in_method or line[0] in ".:#":
                        continue
                    head, _, rest = line.partition(" ")
                    # Payload data (.array-data, switches) never starts with a mnemonic.
                    if head in opcodes:
                        yield parse_instruction(head, rest.strip())


def method_path(op2, max_depth):
    """Tree nodes of an operand: package parts, class, member, signature.

    'Ljava/lang/Object;-><init>()V' -> ['java', 'lang', 'Object', '<init>', '()V']
    """
    if op2.startswith("<"):
        return [op2]
    owner, arrow, member = op2.partition("->")
    nodes = [x for x in owner.lstrip("[").lstrip("L").rstrip(";").split("/") if x]
    if arrow:
        name, paren, sig = member.partition("(")
        name, colon, ftype = name.partition(":")
        nodes.append(name)
        if paren:
            nodes.append("(" + sig)
        elif colon:
            nodes.append(":" + ftype)
    return nodes[:max_depth] or [NONE]


def coarse_operand(op2, depth=2):
    """Short form of op2 used inside 3-grams, to keep their vocabulary bounded."""
    return "/".join(method_path(op2, depth))
