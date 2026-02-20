"""TypeScript / TSX / JavaScript parser using tree-sitter.

Extracts symbols (functions, classes, methods, interfaces, type aliases),
imports, call expressions, type annotation references, and heritage
(extends / implements) relationships from TypeScript, TSX, and JavaScript
source files.
"""

from __future__ import annotations

import tree_sitter_javascript as tsjavascript
import tree_sitter_typescript as tstypescript
from tree_sitter import Language, Node, Parser

from axon.core.parsers.base import (
    CallInfo,
    ImportInfo,
    LanguageParser,
    ParseResult,
    SymbolInfo,
    TypeRef,
)

# ---------------------------------------------------------------------------
# Language singletons
# ---------------------------------------------------------------------------

TS_LANGUAGE = Language(tstypescript.language_typescript())
TSX_LANGUAGE = Language(tstypescript.language_tsx())
JS_LANGUAGE = Language(tsjavascript.language())

_DIALECT_MAP: dict[str, Language] = {
    "typescript": TS_LANGUAGE,
    "tsx": TSX_LANGUAGE,
    "javascript": JS_LANGUAGE,
}

# Built-in TypeScript types that should be skipped in type_refs.
_BUILTIN_TYPES: frozenset[str] = frozenset(
    {
        "string",
        "number",
        "boolean",
        "void",
        "any",
        "unknown",
        "never",
        "null",
        "undefined",
        "object",
    }
)


class TypeScriptParser(LanguageParser):
    """Parse TypeScript, TSX, or JavaScript files via tree-sitter.

    Args:
        dialect: One of ``"typescript"``, ``"tsx"``, or ``"javascript"``.
    """

    def __init__(self, dialect: str = "typescript") -> None:
        if dialect not in _DIALECT_MAP:
            raise ValueError(
                f"Unknown dialect {dialect!r}. "
                f"Expected one of: {', '.join(sorted(_DIALECT_MAP))}"
            )
        self.dialect = dialect
        self._language = _DIALECT_MAP[dialect]
        self._parser = Parser(self._language)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, content: str, file_path: str) -> ParseResult:
        """Parse *content* and return an intermediate :class:`ParseResult`."""
        tree = self._parser.parse(content.encode("utf-8"))

        result = ParseResult()
        self._walk(tree.root_node, content, result)
        return result

    # ------------------------------------------------------------------
    # Recursive walk
    # ------------------------------------------------------------------

    def _walk(
        self, node: Node, source: str, result: ParseResult, visited: set[int] | None = None
    ) -> None:
        """Walk the tree recursively, dispatching on node type.

        Uses a *visited* set (keyed by node ``id``) to avoid processing
        the same subtree twice — e.g. class bodies that are walked by both
        ``_extract_class`` and the generic child recursion.
        """
        if visited is None:
            visited = set()

        node_key = node.id
        if node_key in visited:
            return
        visited.add(node_key)

        ntype = node.type

        if ntype == "function_declaration":
            self._extract_function_declaration(node, source, result)
        elif ntype in ("lexical_declaration", "variable_declaration"):
            self._extract_variable_declaration(node, source, result)
        elif ntype == "class_declaration":
            self._extract_class(node, source, result)
        elif ntype == "interface_declaration":
            self._extract_interface(node, source, result)
        elif ntype == "type_alias_declaration":
            self._extract_type_alias(node, source, result)
        elif ntype == "import_statement":
            self._extract_import(node, source, result)
        elif ntype == "call_expression":
            self._extract_call(node, source, result)
        elif ntype == "method_definition":
            self._extract_method(node, source, result)

        for child in node.children:
            self._walk(child, source, result, visited)

    # ------------------------------------------------------------------
    # Functions
    # ------------------------------------------------------------------

    def _extract_function_declaration(
        self, node: Node, source: str, result: ParseResult
    ) -> None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return

        name = name_node.text.decode()
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        content = node.text.decode()
        signature = self._build_function_signature(node, name)

        result.symbols.append(
            SymbolInfo(
                name=name,
                kind="function",
                start_line=start_line,
                end_line=end_line,
                content=content,
                signature=signature,
            )
        )

        # Extract parameter types and return type.
        self._extract_function_types(node, name, result)

    def _extract_method(self, node: Node, source: str, result: ParseResult) -> None:
        """Extract a method_definition inside a class body."""
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return

        name = name_node.text.decode()
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        content = node.text.decode()

        # Determine owning class by walking up to class_declaration.
        class_name = self._find_parent_class_name(node)

        signature = self._build_function_signature(node, name)

        result.symbols.append(
            SymbolInfo(
                name=name,
                kind="method",
                start_line=start_line,
                end_line=end_line,
                content=content,
                signature=signature,
                class_name=class_name,
            )
        )

        self._extract_function_types(node, name, result)

    # ------------------------------------------------------------------
    # Variable declarations (arrow functions, function expressions, require)
    # ------------------------------------------------------------------

    def _extract_variable_declaration(
        self, node: Node, source: str, result: ParseResult
    ) -> None:
        """Handle arrow functions, function expressions, and require() calls."""
        for child in node.children:
            if child.type != "variable_declarator":
                continue

            name_node = child.child_by_field_name("name")
            value_node = child.child_by_field_name("value")
            if name_node is None or value_node is None:
                continue

            var_name = name_node.text.decode()

            if value_node.type in ("arrow_function", "function_expression"):
                self._extract_assigned_function(child, var_name, value_node, result)
            elif value_node.type == "call_expression":
                self._maybe_extract_require(child, var_name, value_node, result)

            # Variable type annotation: const x: Config = ...
            self._extract_variable_type_annotation(child, result)

    def _extract_assigned_function(
        self,
        declarator_node: Node,
        name: str,
        func_node: Node,
        result: ParseResult,
    ) -> None:
        """Extract an arrow function or function expression assigned to a variable."""
        # Use the entire lexical_declaration as the span.
        outer = declarator_node.parent
        if outer is None:
            outer = declarator_node

        start_line = outer.start_point[0] + 1
        end_line = outer.end_point[0] + 1
        content = outer.text.decode()
        signature = self._build_function_signature(func_node, name)

        result.symbols.append(
            SymbolInfo(
                name=name,
                kind="function",
                start_line=start_line,
                end_line=end_line,
                content=content,
                signature=signature,
            )
        )

        self._extract_function_types(func_node, name, result)

    def _maybe_extract_require(
        self,
        declarator_node: Node,
        var_name: str,
        call_node: Node,
        result: ParseResult,
    ) -> None:
        """If the call is ``require('./foo')``, emit an ImportInfo."""
        func_node = call_node.child_by_field_name("function")
        if func_node is None or func_node.text.decode() != "require":
            return

        args = call_node.child_by_field_name("arguments")
        if args is None:
            return

        # First real argument (skip parentheses).
        module_str = ""
        for arg_child in args.children:
            if arg_child.type == "string":
                module_str = self._string_value(arg_child)
                break

        if not module_str:
            return

        result.imports.append(
            ImportInfo(
                module=module_str,
                names=[var_name],
                is_relative=module_str.startswith("."),
            )
        )

    # ------------------------------------------------------------------
    # Classes
    # ------------------------------------------------------------------

    def _extract_class(self, node: Node, source: str, result: ParseResult) -> None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return

        name = name_node.text.decode()
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        content = node.text.decode()

        result.symbols.append(
            SymbolInfo(
                name=name,
                kind="class",
                start_line=start_line,
                end_line=end_line,
                content=content,
            )
        )

        # Heritage: extends / implements
        for child in node.children:
            if child.type == "class_heritage":
                self._extract_class_heritage(name, child, result)

    def _extract_class_heritage(
        self, class_name: str, heritage_node: Node, result: ParseResult
    ) -> None:
        for child in heritage_node.children:
            if child.type == "extends_clause":
                for sub in child.children:
                    if sub.type in ("identifier", "type_identifier"):
                        result.heritage.append((class_name, "extends", sub.text.decode()))
            elif child.type == "implements_clause":
                for sub in child.children:
                    if sub.type in ("identifier", "type_identifier"):
                        result.heritage.append((class_name, "implements", sub.text.decode()))

    # ------------------------------------------------------------------
    # Interfaces (TypeScript only)
    # ------------------------------------------------------------------

    def _extract_interface(self, node: Node, source: str, result: ParseResult) -> None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return

        name = name_node.text.decode()
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        content = node.text.decode()

        result.symbols.append(
            SymbolInfo(
                name=name,
                kind="interface",
                start_line=start_line,
                end_line=end_line,
                content=content,
            )
        )

        # Interface heritage: extends (interfaces cannot use implements).
        for child in node.children:
            if child.type == "extends_type_clause":
                for sub in child.children:
                    if sub.type in ("identifier", "type_identifier"):
                        result.heritage.append((name, "extends", sub.text.decode()))

    # ------------------------------------------------------------------
    # Type aliases (TypeScript only)
    # ------------------------------------------------------------------

    def _extract_type_alias(self, node: Node, source: str, result: ParseResult) -> None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return

        name = name_node.text.decode()
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        content = node.text.decode()

        result.symbols.append(
            SymbolInfo(
                name=name,
                kind="type_alias",
                start_line=start_line,
                end_line=end_line,
                content=content,
            )
        )

    # ------------------------------------------------------------------
    # Imports
    # ------------------------------------------------------------------

    def _extract_import(self, node: Node, source: str, result: ParseResult) -> None:
        """Handle ES module import statements."""
        module_str = ""
        names: list[str] = []
        alias = ""

        # Find the source/module string.
        source_node = node.child_by_field_name("source")
        if source_node is not None:
            module_str = self._string_value(source_node)
        else:
            # Fallback: look for a string child after 'from'.
            for child in node.children:
                if child.type == "string":
                    module_str = self._string_value(child)
                    break

        if not module_str:
            return

        # Find the import_clause (the part between `import` and `from`).
        import_clause = None
        for child in node.children:
            if child.type == "import_clause":
                import_clause = child
                break

        if import_clause is not None:
            for clause_child in import_clause.children:
                if clause_child.type == "named_imports":
                    # import { A, B } from '...'
                    for spec in clause_child.children:
                        if spec.type == "import_specifier":
                            name_node = spec.child_by_field_name("name")
                            if name_node is not None:
                                names.append(name_node.text.decode())
                elif clause_child.type == "namespace_import":
                    # import * as utils from '...'
                    for ns_child in clause_child.children:
                        if ns_child.type == "identifier":
                            alias = ns_child.text.decode()
                            names.append(alias)
                            break
                elif clause_child.type == "identifier":
                    # import Foo from '...'  (default import)
                    names.append(clause_child.text.decode())

        result.imports.append(
            ImportInfo(
                module=module_str,
                names=names,
                is_relative=module_str.startswith("."),
                alias=alias,
            )
        )

    # ------------------------------------------------------------------
    # Calls
    # ------------------------------------------------------------------

    def _extract_call(self, node: Node, source: str, result: ParseResult) -> None:
        func_node = node.child_by_field_name("function")
        if func_node is None:
            return

        line = node.start_point[0] + 1

        if func_node.type == "member_expression":
            obj_node = func_node.child_by_field_name("object")
            prop_node = func_node.child_by_field_name("property")
            if prop_node is not None:
                receiver = obj_node.text.decode() if obj_node else ""
                result.calls.append(
                    CallInfo(
                        name=prop_node.text.decode(),
                        line=line,
                        receiver=receiver,
                    )
                )
        elif func_node.type == "identifier":
            name = func_node.text.decode()
            # Skip require() since it's handled as an import.
            if name != "require":
                result.calls.append(CallInfo(name=name, line=line))

    # ------------------------------------------------------------------
    # Type annotations
    # ------------------------------------------------------------------

    def _extract_function_types(
        self, func_node: Node, func_name: str, result: ParseResult
    ) -> None:
        """Extract parameter types and return type from a function-like node."""
        # Parameters
        params = func_node.child_by_field_name("parameters")
        if params is None:
            # Some nodes use "formal_parameters" via children iteration.
            for child in func_node.children:
                if child.type == "formal_parameters":
                    params = child
                    break

        if params is not None:
            for param in params.children:
                if param.type in ("required_parameter", "optional_parameter"):
                    param_name_node = param.child_by_field_name("name")
                    if param_name_node is None:
                        # Fallback: first identifier child.
                        for sub in param.children:
                            if sub.type == "identifier":
                                param_name_node = sub
                                break
                    if param_name_node is None:
                        continue

                    param_name = param_name_node.text.decode()

                    for sub in param.children:
                        if sub.type == "type_annotation":
                            type_name = self._type_annotation_name(sub)
                            if type_name and type_name.lower() not in _BUILTIN_TYPES:
                                result.type_refs.append(
                                    TypeRef(
                                        name=type_name,
                                        kind="param",
                                        line=sub.start_point[0] + 1,
                                        param_name=param_name,
                                    )
                                )

        # Return type: type_annotation directly on the function node (not inside params).
        for child in func_node.children:
            if child.type == "type_annotation":
                type_name = self._type_annotation_name(child)
                if type_name and type_name.lower() not in _BUILTIN_TYPES:
                    result.type_refs.append(
                        TypeRef(
                            name=type_name,
                            kind="return",
                            line=child.start_point[0] + 1,
                        )
                    )

    def _extract_variable_type_annotation(
        self, declarator_node: Node, result: ParseResult
    ) -> None:
        """Extract type from ``const x: Config = ...``."""
        for child in declarator_node.children:
            if child.type == "type_annotation":
                type_name = self._type_annotation_name(child)
                if type_name and type_name.lower() not in _BUILTIN_TYPES:
                    result.type_refs.append(
                        TypeRef(
                            name=type_name,
                            kind="variable",
                            line=child.start_point[0] + 1,
                        )
                    )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _type_annotation_name(annotation_node: Node) -> str:
        """Return the simple type name from a ``type_annotation`` node.

        Handles ``type_identifier``, ``predefined_type``, and ``identifier``
        children.  For compound types (unions, generics, etc.) returns the
        text of the first recognisable child.
        """
        for child in annotation_node.children:
            if child.type in ("type_identifier", "predefined_type", "identifier"):
                return child.text.decode()
        return ""

    @staticmethod
    def _string_value(string_node: Node) -> str:
        """Extract the raw string value from a tree-sitter ``string`` node.

        String nodes look like: string -> [quote, string_fragment, quote].
        """
        for child in string_node.children:
            if child.type == "string_fragment":
                return child.text.decode()
        # Fallback: strip outer quotes from the whole text.
        text = string_node.text.decode()
        if len(text) >= 2 and text[0] in ("'", '"', "`") and text[-1] in ("'", '"', "`"):
            return text[1:-1]
        return text

    @staticmethod
    def _build_function_signature(node: Node, name: str) -> str:
        """Build a human-readable signature line for a function-like node.

        Includes the parameter list and return type (if present).
        """
        params_text = ""
        return_type = ""

        for child in node.children:
            if child.type == "formal_parameters":
                params_text = child.text.decode()
            elif child.type == "type_annotation":
                return_type = child.text.decode()

        sig = f"{name}{params_text}"
        if return_type:
            sig += return_type
        return sig

    @staticmethod
    def _find_parent_class_name(node: Node) -> str:
        """Walk up the tree to find the enclosing class name."""
        current = node.parent
        while current is not None:
            if current.type == "class_declaration":
                name_node = current.child_by_field_name("name")
                if name_node is not None:
                    return name_node.text.decode()
            current = current.parent
        return ""
