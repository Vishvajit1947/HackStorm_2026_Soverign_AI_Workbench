"""What language a snippet is, what file it belongs in, and how it is run.

`code.exec` used to write every snippet to `/sandbox/main.py` and hand it to the
Python interpreter whatever it contained. A C program went in and came back as

    File "/sandbox/main.py", line 1
    printf("Hello, World!\\n");
    NameError: name 'printf' is not defined. Did you mean: 'print'?

which is the house rule broken in the worst direction: a Python interpreter
reporting a Python error over a file that was never Python, so the transcript
blames the model for code that is perfectly good C. The reader has to know the
sandbox is Python-only to decode that message, and nothing on screen says so.

One table drives everything: the extension the source is written with, the
binaries that have to exist, the compile step if there is one, and the run step.
Docker and the development fallback both build their commands from it, so the
two runners cannot disagree about what "run some Java" means.

Two safety properties hold this together, and both must survive any edit:

  NOTHING MODEL-AUTHORED IS EVER INTERPOLATED INTO A SHELL STRING. A compiled
  language needs `compile && run`, which needs a shell. Every fragment that
  reaches it comes from this table, from a path this module chose, or from a
  Java class name that matched `_JAVA_IDENT` - never from the snippet. The
  snippet only ever arrives as the CONTENTS of a file.

  A MISSING TOOLCHAIN IS REPORTED, NEVER SUBSTITUTED. If the image has no
  `javac`, the answer is "this sandbox image has no Java toolchain" plus the
  line that fixes it - not a run of something else that happens to be present.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field


class UnknownLanguage(ValueError):
    """A language name nothing in this table answers to."""


@dataclass(frozen=True)
class Language:
    """One language, described once for both runners.

    `compile` and `run` are argv templates. The placeholders are filled by the
    runner because their values differ between a container and the host:

        {src}    the staged source file
        {out}    the compiled executable to produce
        {dir}    a writable directory for build output
        {class}  the Java class to invoke (see `source_filename`)
    """

    id: str
    label: str
    extension: str
    binaries: tuple[str, ...]
    run: tuple[str, ...]
    compile: tuple[str, ...] | None = None
    # Extra environment the toolchain needs on top of the sandbox's own. Go
    # refuses to build without a writable cache, and the sandbox root filesystem
    # is read-only by design, so it has to be pointed at the tmpfs.
    env: dict[str, str] = field(default_factory=dict)
    # Said out loud when the toolchain is absent, because "install it" is not
    # actionable on an air-gapped box - the image is what has to change.
    install_hint: str = ""
    # A fixed shell fragment run before the compile, from THIS TABLE and never
    # from a snippet. For toolchains that need a file in place rather than an
    # environment variable.
    prelude: str = ""
    # Set when the toolchain is INSTALLED and still cannot run here. The call is
    # refused up front naming this, rather than letting someone write a program
    # and discover it from a linker error at the end - which is the same rule
    # that makes a missing toolchain a refusal rather than a wrong interpreter.
    unsupported: str = ""

    @property
    def compiled(self) -> bool:
        return self.compile is not None


# The binaries named here are Debian's, because the sandbox image is
# python:3.12-slim. `gcc`/`g++` rather than the `cc`/`c++` alternatives: an
# alternative can point somewhere else, and the error when it does is confusing.
LANGUAGES: dict[str, Language] = {
    "python": Language(
        id="python",
        label="Python",
        extension=".py",
        binaries=("python",),
        run=("python", "{src}"),
        install_hint="python is the sandbox image's base; if it is missing the image is broken",
    ),
    "javascript": Language(
        id="javascript",
        label="JavaScript (Node)",
        extension=".js",
        binaries=("node",),
        run=("node", "{src}"),
        install_hint="add `nodejs` to the apt install line in deploy/Dockerfile.sandbox",
    ),
    "c": Language(
        id="c",
        label="C",
        extension=".c",
        binaries=("gcc",),
        # -lm last: a plant calculation reaches for sqrt/pow within two lines,
        # and glibc keeps them in a library the linker needs told about.
        compile=("gcc", "-O2", "-std=c17", "-o", "{out}", "{src}", "-lm"),
        run=("{out}",),
        install_hint="add `gcc libc6-dev` to the apt install line in deploy/Dockerfile.sandbox",
    ),
    "cpp": Language(
        id="cpp",
        label="C++",
        extension=".cpp",
        binaries=("g++",),
        compile=("g++", "-O2", "-std=c++17", "-o", "{out}", "{src}"),
        run=("{out}",),
        install_hint="add `g++` to the apt install line in deploy/Dockerfile.sandbox",
    ),
    "java": Language(
        id="java",
        label="Java",
        extension=".java",
        binaries=("javac", "java"),
        compile=("javac", "-d", "{dir}", "{src}"),
        run=("java", "-cp", "{dir}", "{class}"),
        install_hint="add `default-jdk-headless` to the apt install line in deploy/Dockerfile.sandbox",
    ),
    "shell": Language(
        id="shell",
        label="Shell (sh)",
        extension=".sh",
        binaries=("sh",),
        run=("sh", "{src}"),
        install_hint="every image has a shell; if sh is missing the image is broken",
    ),
    "go": Language(
        id="go",
        label="Go",
        extension=".go",
        binaries=("go",),
        run=("go", "run", "{src}"),
        # GOTELEMETRY is not optional here. Go 1.24 forks a telemetry child on
        # every invocation, and in a container with a read-only rootfs and
        # cap-drop ALL that fork fails - so a correct Go program exits 1 with
        # "can't start telemetry child process" and nothing about the program.
        # The sandbox has no network either way; this just stops it trying.
        env={
            "GOCACHE": "{dir}/.gocache", "GOPATH": "{dir}/.go",
            "GOFLAGS": "-mod=mod", "GOTOOLCHAIN": "local",
            # /tmp is on the read-only rootfs. Without this `go run` fails with
            # "creating work dir: mkdir /tmp/go-build...: read-only file system"
            # before it ever looks at the program.
            "TMPDIR": "{dir}", "HOME": "{dir}",
        },
        # Go 1.24 forks a telemetry child on every invocation and prints
        # "can't start telemetry child process" to stderr when it cannot - which
        # it never can here, and which looks like an error in the program. There
        # is no environment variable for it; the mode file is the control.
        prelude=(
            'mkdir -p "$HOME/.config/go/telemetry" 2>/dev/null; '
            'printf off > "$HOME/.config/go/telemetry/mode" 2>/dev/null; '
        ),
        install_hint="add `golang-go` to the apt install line in deploy/Dockerfile.sandbox",
    ),
    "rust": Language(
        id="rust",
        label="Rust",
        extension=".rs",
        binaries=("rustc",),
        compile=("rustc", "-O", "-o", "{out}", "{src}"),
        run=("{out}",),
        # rustc writes intermediates to TMPDIR, which is /tmp - and /tmp is on
        # the read-only rootfs. Without this a correct Rust program fails with
        # "couldn't create a temp dir: Read-only file system (os error 30)",
        # which reads as a broken sandbox rather than a missing variable.
        env={"TMPDIR": "{dir}", "HOME": "{dir}"},
        # MEASURED, not assumed: rustc compiles fine and then cannot exec its
        # linker - "could not exec the linker `cc`: Operation not permitted" -
        # under the strict seccomp profile. Allowing fork, execveat, setpgid,
        # clone3, personality, setsid and the sched_set* family one at a time
        # did not move it; removing the profile entirely does. Rather than let
        # somebody write a Rust program and meet that at the end, the call is
        # refused at the start. Whoever identifies the syscall should add it to
        # deploy/seccomp-strict.json and delete this line - the rest of the Rust
        # path is correct and tested.
        unsupported=(
            "rustc runs in this image but cannot exec its linker under the "
            "sandbox's seccomp profile, so no Rust program can be linked here. "
            "The other eight languages are unaffected. Fix: identify the syscall "
            "rustc's linker step needs and allow it in deploy/seccomp-strict.json"
        ),
        install_hint="add `rustc` to the apt install line in deploy/Dockerfile.sandbox",
    ),
    "ruby": Language(
        id="ruby",
        label="Ruby",
        extension=".rb",
        binaries=("ruby",),
        run=("ruby", "{src}"),
        install_hint="add `ruby` to the apt install line in deploy/Dockerfile.sandbox",
    ),
}

# What a model actually types. `c++`, `py` and `node` are not obscure - they are
# the first word most models reach for - and a tool that refuses them teaches the
# planner nothing except that the argument is a trap.
ALIASES: dict[str, str] = {
    "py": "python", "python3": "python", "py3": "python",
    "js": "javascript", "node": "javascript", "nodejs": "javascript",
    "ecmascript": "javascript", "mjs": "javascript",
    "c++": "cpp", "cxx": "cpp", "cc": "cpp", "g++": "cpp",
    "gcc": "c", "clang": "c", "ansi-c": "c",
    "jdk": "java", "javac": "java",
    "sh": "shell", "bash": "shell", "zsh": "shell", "posix-shell": "shell",
    "golang": "go",
    "rs": "rust", "rustlang": "rust",
    "rb": "ruby",
}

# Suffix -> language, so a `files` entry or an fs.write path can say what a
# snippet is without the model naming it twice.
BY_EXTENSION: dict[str, str] = {
    **{lang.extension: lang.id for lang in LANGUAGES.values()},
    ".pyw": "python",
    ".cxx": "cpp", ".cc": "cpp", ".hpp": "cpp", ".hh": "cpp",
    ".h": "c",
    ".bash": "shell",
    ".mjs": "javascript", ".cjs": "javascript",
}


def resolve(name: str) -> Language:
    """A language by any of the names it is actually called.

    Raises rather than falling back to Python: a fallback here is exactly the
    bug this module exists to kill.
    """
    key = (name or "").strip().lower().lstrip(".")
    key = ALIASES.get(key, key)
    if key in LANGUAGES:
        return LANGUAGES[key]
    if f".{key}" in BY_EXTENSION:
        return LANGUAGES[BY_EXTENSION[f".{key}"]]
    raise UnknownLanguage(
        f"'{name}' is not a language code.exec knows. Supported: "
        + ", ".join(sorted(LANGUAGES)) + "."
    )


def language_for_path(path: str) -> Language | None:
    """The language of a filename, or None. Never guesses from the stem."""
    match = re.search(r"(\.[A-Za-z0-9+]+)$", (path or "").strip())
    if not match:
        return None
    lang_id = BY_EXTENSION.get(match.group(1).lower())
    return LANGUAGES.get(lang_id) if lang_id else None


# ------------------------------------------------------------------ fences

_FENCE = re.compile(
    r"\A\s*```[ \t]*([A-Za-z0-9_+#.-]*)[ \t]*\r?\n(.*?)\r?\n?[ \t]*```\s*\Z",
    re.DOTALL,
)


def strip_fence(code: str) -> tuple[str, str | None]:
    """(source, language named on the fence).

    Models wrap snippets in ```c ... ``` constantly, and the fence is the one
    place the author SAID what the code is. Passing it through to a compiler
    turns a correct program into a syntax error on line 1, and throwing it away
    loses the most reliable signal available.
    """
    match = _FENCE.match(code or "")
    if not match:
        return code, None
    hint = match.group(1).strip().lower() or None
    return match.group(2), hint


# ------------------------------------------------------------------ detection

_JAVA_IDENT = re.compile(r"\A[A-Za-z_$][A-Za-z0-9_$]*\Z")
_PUBLIC_CLASS = re.compile(r"^\s*public\s+(?:final\s+|abstract\s+)?class\s+([A-Za-z_$][\w$]*)", re.MULTILINE)
_ANY_CLASS = re.compile(r"^\s*(?:final\s+|abstract\s+)?class\s+([A-Za-z_$][\w$]*)", re.MULTILINE)

# Ordered, most specific first. Each rule is a (language, pattern, why) and the
# `why` is carried into the result: a reader who disagrees with the detection
# needs to see what it keyed on, not just what it decided.
_RULES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    # A shebang is the author saying outright what the file is, so it outranks
    # every heuristic under it.
    ("shell", re.compile(r"\A#!\s*\S*\b(?:ba|z|k|da)?sh\b"), "a shell shebang"),
    ("python", re.compile(r"\A#!\s*\S*\bpython[0-9.]*\b"), "a python shebang"),
    ("java", re.compile(r"\bpublic\s+static\s+void\s+main\s*\("), "a `public static void main` entry point"),
    ("java", re.compile(r"\bSystem\.out\.print"), "a System.out.print call"),
    ("java", re.compile(r"^\s*import\s+java\.", re.MULTILINE), "an import of java.*"),
    ("go", re.compile(r"^\s*package\s+main\b", re.MULTILINE), "a `package main` declaration"),
    ("go", re.compile(r"\bfunc\s+main\s*\(\s*\)\s*\{"), "a `func main()` entry point"),
    ("go", re.compile(r"^\s*import\s+\"fmt\"", re.MULTILINE), "an import of Go's fmt"),
    ("rust", re.compile(r"\bfn\s+main\s*\(\s*\)"), "an `fn main()` entry point"),
    ("rust", re.compile(r"\b(?:println!|let\s+mut\s)"), "Rust's println! / let mut"),
    ("cpp", re.compile(r"#include\s*<(?:iostream|vector|string|map|algorithm)>"), "a C++ standard header"),
    ("cpp", re.compile(r"\bstd::"), "a std:: qualified name"),
    ("cpp", re.compile(r"\busing\s+namespace\s+std\b"), "`using namespace std`"),
    ("c", re.compile(r"#include\s*<[A-Za-z0-9_/.]+\.h>"), "a C header include"),
    # `printf(` on its own, deliberately: the bug that started all of this was a
    # one-line `printf("Hello, World!\n");` with no include and no main, which
    # any rule wanting a whole C program would still have handed to Python.
    # `printf` is not a Python name, so there is nothing for it to belong to.
    ("c", re.compile(r"\bprintf\s*\("), "a printf call"),
    ("c", re.compile(r"\b(?:scanf|malloc|fprintf|sizeof)\s*\("), "a C standard-library call"),
    ("c", re.compile(r"\bint\s+main\s*\("), "an `int main(...)` entry point"),
    # Ruby is keyed on `puts` alone. A rule for its `end` terminator was tried
    # and removed: `end` appears inside string literals, which detection keeps,
    # so a Python program printing "the end" detected as Ruby.
    ("ruby", re.compile(r"^\s*puts\s+", re.MULTILINE), "a `puts` statement"),
    ("javascript", re.compile(r"\bconsole\.(?:log|error|warn)\s*\("), "a console.log call"),
    ("javascript", re.compile(r"^\s*(?:const|let|var)\s+\w+\s*=", re.MULTILINE), "a const/let/var declaration"),
    ("javascript", re.compile(r"\b(?:function\s*\w*\s*\(|=>\s*\{)"), "a JavaScript function form"),
    ("javascript", re.compile(r"\brequire\s*\(\s*['\"]"), "a require() call"),
    ("python", re.compile(r"^\s*(?:def|class)\s+\w+.*:\s*$", re.MULTILINE), "a def/class header ending in a colon"),
    ("python", re.compile(r"^\s*(?:from\s+[\w.]+\s+)?import\s+\w", re.MULTILINE), "a Python import"),
    ("python", re.compile(r"\bprint\s*\((?!.*\)\s*;)"), "a print() call with no trailing semicolon"),
    ("python", re.compile(r"^\s*if\s+__name__\s*==", re.MULTILINE), "an `if __name__ ==` guard"),
)

# Weighed against everything else before Python wins by default: `printf(...);`
# is C in any language that has both, and the original bug was Python claiming
# a file it could not parse.
_DEFAULT = "python"


def detect(code: str) -> tuple[Language, str]:
    """(language, the evidence for it). Never raises - there is always a default.

    The default is Python because that is what the sandbox has always been and
    what most snippets are. What changed is that the default is now the LAST
    answer rather than the only one, and the caller is told which it got.
    """
    source, hint = strip_fence(code)
    if hint:
        try:
            return resolve(hint), f"the ```{hint} fence the snippet was written in"
        except UnknownLanguage:
            pass                      # an unknown fence is not a reason to fail

    stripped = _without_comments_and_strings(source)
    for lang_id, pattern, why in _RULES:
        if pattern.search(stripped):
            return LANGUAGES[lang_id], why
    return LANGUAGES[_DEFAULT], "no language marker was found, so the sandbox default was used"


# `#` opens a comment in Python, Ruby and shell - and a preprocessor directive
# in C and C++. Stripping it blindly deletes `#include <stdio.h>`, which is the
# single strongest C signal there is, and the file then detects as Python again.
_LINE_COMMENT = re.compile(
    r"(?m)(?://.*$|\#(?!\s*(?:include|define|ifndef|ifdef|endif|pragma|undef|elif|else|if|error)\b).*$)"
)
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_TRIPLE = re.compile(r"'''.*?'''|\"\"\".*?\"\"\"", re.DOTALL)


def _without_comments_and_strings(code: str) -> str:
    """Detection reads the code, not the prose around it.

    A Python file whose docstring says "this is the C version of..." must not
    detect as C, and a C file commented `// printf is like print` must not lose
    its printf. Only comment bodies are removed; string literals are kept,
    because `System.out.println("hi")` is evidence and its argument is harmless.
    """
    text = _TRIPLE.sub(" ", code)
    text = _BLOCK_COMMENT.sub(" ", text)
    shebang = ""
    if text.startswith("#!"):
        shebang, _, text = text.partition("\n")
        shebang += "\n"
    return shebang + _LINE_COMMENT.sub("", text)


# ------------------------------------------------------------------ filenames

def source_filename(language: Language, code: str) -> str:
    """What the snippet has to be called for its own toolchain to accept it.

    Java is the reason this is a function. `javac` refuses a public class whose
    file name does not match it, so a model writing `public class HelloWorld`
    gets "class HelloWorld is public, should be declared in a file named
    HelloWorld.java" - a compiler error caused entirely by us, about code that
    is correct.

    The name is validated against `_JAVA_IDENT` before it is used, because it
    reaches the run command as the class to invoke. Anything else falls back to
    `Main` rather than being sanitised into something the source does not
    declare.
    """
    if language.id != "java":
        return f"main{language.extension}"
    return f"{java_class(code)}.java"


def java_class(code: str) -> str:
    source, _ = strip_fence(code)
    source = _without_comments_and_strings(source)
    for pattern in (_PUBLIC_CLASS, _ANY_CLASS):
        match = pattern.search(source)
        if match and _JAVA_IDENT.match(match.group(1)):
            return match.group(1)
    return "Main"


# ------------------------------------------------------------------ commands

_ENTRY_POINT = {
    "c": (re.compile(r"\bmain\s*\("), "int main(void) { ... }"),
    "cpp": (re.compile(r"\bmain\s*\("), "int main() { ... }"),
    "java": (re.compile(r"\bstatic\s+void\s+main\s*\("),
             "public static void main(String[] args) { ... }"),
    "rust": (re.compile(r"\bfn\s+main\s*\("), "fn main() { ... }"),
    "go": (re.compile(r"\bfunc\s+main\s*\("), "func main() { ... }"),
}


def missing_entry_point(language: Language, code: str) -> str | None:
    """The entry point a compiled snippet needs and does not have, or None.

    A model often writes the one line that does the work - `printf("hi\\n");` -
    which is a statement, not a program, and gcc says so in terms that assume
    the reader knows C. This only ever produces a HINT alongside the compiler's
    own message: wrapping the snippet in a main() nobody wrote would put lines
    in the Code tab that the model did not write and the reader cannot audit.
    """
    pattern = _ENTRY_POINT.get(language.id)
    if pattern is None:
        return None
    source, _ = strip_fence(code)
    if pattern[0].search(_without_comments_and_strings(source)):
        return None
    return pattern[1]


# The header each of these lives in. Modern gcc makes an implicit declaration
# an ERROR under -std=c17, so a forgotten `#include <stdio.h>` - which is the
# single most common thing a small model leaves out of a C program - fails the
# compile with a message about "implicit declaration" that says nothing about
# the missing line.
_C_HEADERS = {
    "stdio.h": ("printf", "scanf", "fprintf", "sprintf", "puts", "fopen", "fgets"),
    "stdlib.h": ("malloc", "calloc", "free", "atoi", "atof", "exit"),
    "math.h": ("sqrt", "pow", "fabs", "sin", "cos", "tan", "log", "exp"),
    "string.h": ("strlen", "strcpy", "strcmp", "strcat", "memset", "memcpy"),
}


def missing_header(language: Language, code: str) -> str | None:
    """The `#include` this C/C++ snippet calls into and never included.

    A HINT beside the compiler's own message, never an edit: inserting the line
    would put source in the file that the model did not write and the reader
    cannot audit, which is the same rule `missing_entry_point` follows.
    """
    if language.id not in ("c", "cpp"):
        return None
    source, _ = strip_fence(code)
    body = _without_comments_and_strings(source)
    for header, functions in _C_HEADERS.items():
        if f"<{header}>" in body:
            continue
        for name in functions:
            if re.search(rf"\b{name}\s*\(", body):
                return f"#include <{header}>   (for {name})"
    return None


def fill(argv: tuple[str, ...], **slots: str) -> list[str]:
    return [part.format(**slots) for part in argv]


def shell_command(language: Language, *, src: str, out: str, dir: str, klass: str) -> str:
    """`compile && run` as one POSIX command line, for a container.

    Docker runs one process; a compiled language needs two. Every fragment here
    comes from the table above or from a path this module chose, and each is
    quoted anyway - see the note at the top of the module about what may never
    reach a shell.
    """
    slots = {"src": src, "out": out, "dir": dir, "class": klass}
    run = " ".join(shlex.quote(part) for part in fill(language.run, **slots))
    exports = "".join(
        f"export {key}={shlex.quote(value.format(**slots))}; "
        for key, value in language.env.items()
    )
    prelude = language.prelude.format(**slots) if language.prelude else ""
    if not language.compiled:
        return f"{exports}{prelude}exec {run}"
    compile_line = " ".join(shlex.quote(part) for part in fill(language.compile, **slots))
    return f"{exports}{prelude}{compile_line} && exec {run}"


def summary() -> list[dict[str, object]]:
    """Every language the table knows, for the tool catalogue and the panel."""
    return [
        {
            "id": lang.id,
            "label": lang.label,
            "extension": lang.extension,
            "compiled": lang.compiled,
            "binaries": list(lang.binaries),
        }
        for lang in sorted(LANGUAGES.values(), key=lambda item: item.id)
    ]
