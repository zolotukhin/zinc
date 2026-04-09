//! Shared C import for the Metal shim — all Metal modules import from here
//! to ensure type identity across compilation units.
//!
//! Keeping the `@cImport` in one place avoids duplicate opaque C types across
//! Zig compilation units, which is critical for safely passing shim handles
//! between the Metal device, buffer, pipeline, and command helpers.
/// Raw Metal shim C bindings imported from the Objective-C bridge header.
pub const shim = @cImport(@cInclude("shim.h"));
