// Keep device diagnostics valid JSON without adding a runtime JSON dependency.
// Valid UTF-8 is preserved; malformed bytes become Unicode replacement chars.
#ifndef ZINC_INTEL_PROBE_JSON_H
#define ZINC_INTEL_PROBE_JSON_H
#include <cstddef>
#include <cstdint>
#include <string>

inline std::string intel_probe_json_quote(const std::string &value) {
    const char hex[] = "0123456789abcdef";
    std::string out = "\"";
    for (std::size_t i = 0; i < value.size();) {
        const auto c = static_cast<unsigned char>(value[i]);
        if (c < 128) {
            if (c == '"' || c == '\\') out += '\\';
            if (c < 32 || c == 127) {
                out += "\\u00";
                out += hex[c >> 4];
                out += hex[c & 15];
            } else out += static_cast<char>(c);
            ++i;
            continue;
        }
        const std::size_t length = c >= 0xc2 && c <= 0xdf ? 2 :
            c >= 0xe0 && c <= 0xef ? 3 : c >= 0xf0 && c <= 0xf4 ? 4 : 0;
        bool valid = length != 0 && i + length <= value.size();
        std::uint32_t codepoint = length == 2 ? c & 31 : length == 3 ? c & 15 : c & 7;
        for (std::size_t j = 1; valid && j < length; ++j) {
            auto tail = static_cast<unsigned char>(value[i + j]);
            valid = tail >= 0x80 && tail <= 0xbf;
            codepoint = (codepoint << 6) | (tail & 63);
        }
        valid = valid && (length != 2 || codepoint >= 0x80) &&
            (length != 3 || codepoint >= 0x800) && (length != 4 || codepoint >= 0x10000) &&
            codepoint <= 0x10ffff && !(codepoint >= 0xd800 && codepoint <= 0xdfff);
        if (valid) {
            out.append(value, i, length);
            i += length;
        } else {
            out += "\\ufffd";
            ++i;
        }
    }
    return out + '"';
}
#endif
