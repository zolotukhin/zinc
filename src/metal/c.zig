//! Shared C import for the Metal shim — all Metal modules import from here
//! to ensure type identity across compilation units.
pub const shim = @cImport(@cInclude("shim.h"));
