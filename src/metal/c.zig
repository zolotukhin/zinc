//! Shared C import for the Metal shim — all Metal modules import from here
//! to ensure type identity across compilation units.
/// Raw Metal shim C bindings imported from the Objective-C bridge header.
pub const shim = @cImport(@cInclude("shim.h"));
