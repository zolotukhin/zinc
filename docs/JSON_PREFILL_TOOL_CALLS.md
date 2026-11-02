# JSON-prefilled tool calls

The Intel starter-model sweep exposed three server defects in the generic
ChatML tool path. They are distinct from model coding ability:

- A required single tool prefills its name and first argument in the prompt.
  JSON responses parsed only the generated tail, while SSE rebuilt the call.
- Field selection searched the schema as text. A nested object's `required`
  array could incorrectly select the outer call's first argument.
- Invalid or truncated streaming arguments became an executable empty object.

JSON and SSE now restore the same prefilled prefix and use the ordinary
complete-envelope parser. Incomplete or malformed calls remain text. An empty
argument object is accepted when the model actually completes that object;
it is never substituted for a failed parse. Root schema fields are selected
from parsed JSON, preserving numbers, booleans, arrays, objects and strings.
Temporary schema names and parsed outputs have explicit ownership.

CPU regressions cover nested and escaped schema names, typed arguments,
literal closing-tag text inside strings, complete empty objects, incomplete
strings/objects/envelopes and allocation failures. Live model checks must
record API behavior separately from coding-task outcomes; correcting these
transport defects does not establish that the small model completes a task.
