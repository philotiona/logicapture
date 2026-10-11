# Neuro-Symbolic DSL Vocabulary Specification

This document outlines the complete **Domain-Specific Language (DSL)** vocabulary for the Neuro-Symbolic Visual Rule Discovery System.

---

## 1. Primitive Object Attributes (Unary Predicates)

Evaluates inherent properties of an individual detected object $x$.

| Category | Predicate / Token | Values / Thresholds | Python Implementation Logic |
| :--- | :--- | :--- | :--- |
| **Shape** | `is_shape(x, S)` | `circle`, `square`, `triangle` | `x['shape'] == S` |
| **Color** | `is_color(x, C)` | `red`, `blue`, `green`, `yellow` | `x['color'] == C` |
| **Size** | `is_size(x, Z)` | `small`, `large` | `x['area'] < 1500` vs `x['area'] >= 1500` |

---

## 2. Spatial & Comparative Relations (Binary Predicates)

Evaluates spatial relationships and comparison conditions between two distinct objects $(a, b)$.

| Predicate / Token | Mathematical Definition | Python Condition |
| :--- | :--- | :--- |
| `is_left_of(a, b)` | $x_a < x_b$ | `a['center_x'] < b['center_x']` |
| `is_right_of(a, b)` | $x_a > x_b$ | `a['center_x'] > b['center_x']` |
| `is_above(a, b)` | $y_a < y_b$ *(screen coordinates)* | `a['center_y'] < b['center_y']` |
| `is_below(a, b)` | $y_a > y_b$ *(screen coordinates)* | `a['center_y'] > b['center_y']` |
| `is_near(a, b)` | $\text{EuclideanDist}(a, b) < \tau$ | `math.hypot(a['x'] - b['x'], a['y'] - b['y']) < 60` |
| `is_inside(a, b)` | Bounding box inclusion | `a['xmin'] > b['xmin'] and a['xmax'] < b['xmax']` |
| `same_color(a, b)` | $\text{Color}(a) = \text{Color}(b)$ | `a['color'] == b['color']` |
| `same_shape(a, b)` | $\text{Shape}(a) = \text{Shape}(b)$ | `a['shape'] == b['shape']` |

---

## 3. Set Operations & Quantifiers

Scopes rule evaluations across all objects in a scene $S = \{o_1, o_2, \dots, o_n\}$.

| Operator | Syntax | Description |
| :--- | :--- | :--- |
| **EXISTS** | `EXISTS(x in S, Condition(x))` | Returns `True` if at least 1 object satisfies `Condition`. |
| **FORALL** | `FORALL(x in S, Condition(x))` | Returns `True` if every object in the scene satisfies `Condition`. |
| **COUNT** | `COUNT(x in S, Condition(x)) [op] N` | Counts matching objects. Operator `[op]` $\in \{\text{==}, \ge, \le\}$, where $N \in \{1, 2, 3, 4\}$. |
| **EXISTS_PAIR** | `EXISTS_PAIR(a, b in S, Relation(a, b))` | Returns `True` if there exists any pair $(a, b)$ satisfying `Relation`. |

---

## 4. Logical Operators

Combines predicates into compound expressions.

- **`AND(P, Q)`**: Both conditions must evaluate to `True`.
- **`OR(P, Q)`**: At least one condition must evaluate to `True`.
- **`NOT(P)`**: Inverts the boolean result of condition $P$.
- **`IMPLIES(P, Q)`**: Logical implication (`NOT(P) OR Q`).

---

## 5. Grammar Definition (BNF Format)

```bnf
<Rule>        ::= <Quantifier> | <LogicExpr>
<Quantifier>  ::= "EXISTS(" <Var> ", " <AttrCond> ")"
                | "FORALL(" <Var> ", " <AttrCond> ")"
                | "COUNT(" <Var> ", " <AttrCond> ") " <CompOp> " " <Num>
                | "EXISTS_PAIR(" <Var> "," <Var> ", " <RelCond> ")"

<AttrCond>    ::= <SingleAttr> | "AND(" <SingleAttr> ", " <SingleAttr> ")"
<RelCond>     ::= <SingleRel>  | "AND(" <AttrCond> ", " <SingleRel> ")"

<SingleAttr>  ::= "is_shape(" <Var> ", " <ShapeVal> ")"
                | "is_color(" <Var> ", " <ColorVal> ")"
                | "is_size(" <Var> ", " <SizeVal> ")"

<SingleRel>   ::= "is_left_of(" <Var> ", " <Var> ")"
                | "is_right_of(" <Var> ", " <Var> ")"
                | "is_above(" <Var> ", " <Var> ")"
                | "is_below(" <Var> ", " <Var> ")"
                | "is_near(" <Var> ", " <Var> ")"
                | "is_inside(" <Var> ", " <Var> ")"

<CompOp>      ::= "==" | ">=" | "<="
<Num>         ::= "1" | "2" | "3" | "4"
<ShapeVal>    ::= "circle" | "square" | "triangle"
<ColorVal>    ::= "red" | "blue" | "green" | "yellow"
<SizeVal>     ::= "small" | "large"
<Var>         ::= "x" | "y"