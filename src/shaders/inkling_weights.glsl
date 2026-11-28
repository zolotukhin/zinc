// GGUF weight rows viewed as 32-bit storage so half and byte loads also work
// on Vulkan devices without 8-bit storage.
const uint F32 = 0u, F16 = 1u, Q4_0 = 2u, Q8_0 = 8u;
const uint Q4_K = 12u, Q5_K = 13u, Q6_K = 14u, BF16 = 30u;

uint weight_byte(uint byte_offset) {
    return (weights[byte_offset >> 2u] >> ((byte_offset & 3u) * 8u)) & 255u;
}

float weight_half(uint byte_offset) {
    vec2 pair = unpackHalf2x16(weights[byte_offset >> 2u]);
    return pair[(byte_offset >> 1u) & 1u];
}

uint weight_row_bytes(uint kind, uint cols) {
    switch (kind) {
        case F32: return cols * 4u;
        case F16: case BF16: return cols * 2u;
        case Q4_0: return cols / 32u * 18u;
        case Q8_0: return cols / 32u * 34u;
        case Q4_K: return cols / 256u * 144u;
        case Q5_K: return cols / 256u * 176u;
        case Q6_K: return cols / 256u * 210u;
    }
    return 0u;
}

float weight_value(uint row, uint kind, uint i) {
    switch (kind) {
        case F32: return uintBitsToFloat(weights[(row >> 2u) + i]);
        case F16: return weight_half(row + 2u * i);
        case BF16: {
            uint off = row + 2u * i;
            return uintBitsToFloat((weights[off >> 2u] >> ((off & 2u) * 8u)) << 16u);
        }
        case Q4_0: {
            uint block = row + i / 32u * 18u;
            uint q = weight_byte(block + 2u + (i & 15u)) >> ((i & 16u) >> 2u);
            return weight_half(block) * float(int(q & 15u) - 8);
        }
        case Q8_0: {
            uint block = row + i / 32u * 34u;
            int q = bitfieldExtract(int(weight_byte(block + 2u + (i & 31u))), 0, 8);
            return weight_half(block) * float(q);
        }
        case Q4_K: case Q5_K: {
            bool five = kind == Q5_K;
            uint block = row + i / 256u * (five ? 176u : 144u);
            uint sub = (i & 255u) / 32u;
            uint lane = i & 31u;
            uint scale = block + 4u;
            uint sc = sub < 4u ? weight_byte(scale + sub) & 63u :
                (weight_byte(scale + sub + 4u) & 15u) | ((weight_byte(scale + sub - 4u) >> 6u) << 4u);
            uint mn = sub < 4u ? weight_byte(scale + sub + 4u) & 63u :
                (weight_byte(scale + sub + 4u) >> 4u) | ((weight_byte(scale + sub) >> 6u) << 4u);
            uint packed = weight_byte(block + (five ? 48u : 16u) + (sub / 2u) * 32u + lane);
            uint q = ((packed >> ((sub & 1u) * 4u)) & 15u) |
                (five ? ((weight_byte(block + 16u + lane) >> sub) & 1u) << 4u : 0u);
            return weight_half(block) * float(sc * q) - weight_half(block + 2u) * float(mn);
        }
        case Q6_K: {
            uint block = row + i / 256u * 210u;
            uint half_block = (i & 255u) / 128u;
            uint part = (i & 127u) / 32u;
            uint lane = i & 31u;
            uint lo = weight_byte(block + half_block * 64u + (part & 1u) * 32u + lane);
            uint hi = weight_byte(block + 128u + half_block * 32u + lane);
            int q = int(((lo >> ((part / 2u) * 4u)) & 15u) | (((hi >> (part * 2u)) & 3u) << 4u)) - 32;
            int scale = bitfieldExtract(int(weight_byte(block + 192u + half_block * 8u + part * 2u + lane / 16u)), 0, 8);
            return weight_half(block + 208u) * float(scale * q);
        }
    }
    return 0.0;
}
