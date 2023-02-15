#pragma once

#include <string>
#include <cstdint>
#include <sstream>
#include <iomanip>

struct alignas(16) XmmVal
{
public:
    union {
        uint8_t m_I8[16];
        int16_t m_I16[8];
        int32_t m_I32[4];
        int64_t m_I64[2];
        uint8_t m_U8[16];
        uint16_t m_U16[8];
        uint32_t m_U32[4];
        uint64_t m_U64[2];
        float m_F32[4];
        double m_F64[2];
    };

public:
    // signed integer
    std::string ToStringI8(void)
    {
        return ToStringInt(m_I8, sizeof(m_I8) / sizeof(int8_t), 4);
    }

    std::string ToStringI16(void)
    {
        return ToStringInt(m_I16, sizeof(m_I16) / sizeof(int16_t), 8);
    }

    std::string ToStringI32(void)
    {
        return ToStringInt(m_I32, sizeof(m_I32) / sizeof(int32_t), 16);
    }

    std::string ToStringI64(void)
    {
        return ToStringInt(m_I64, sizeof(m_I64) / sizeof(int64_t) , 32);
    }

    // unsigned integer
    std::string ToStringU8(void)
    {
        return ToStringUint(m_U8, sizeof(m_U8) / sizeof(uint8_t), 4);
    }

    std::string ToStringU16(void)
    {
        return ToStringUint(m_U16, sizeof(m_U16) / sizeof(uint16_t), 8);
    }

    std::string ToStringU32(void)
    {
        return ToStringUint(m_U32, sizeof(m_U32) / sizeof(uint32_t), 16);
    }

    std::string ToStringU64(void)
    {
        return ToStringUint(m_U64, sizeof(m_U64) / sizeof(uint64_t) , 32);
    }

private:
    template <typename  T>
    std::string ToStringInt(const T* x, int n , int w)
    {
        std::ostringstream oss;
        for (int i = 0; i < n; i++) {
            oss << std::setw(w) << (int64_t)x[i];
            if (i + 1 == n / 2) {
                oss << "    |";
            }
        }
        return oss.str();
    }

    template <typename T>
    std::string ToStringUint(const T* x, int n, int w)
    {
        std::ostringstream oss;
        for (int i = 0; i < n; i++) {
            oss << std::setw(w) << (uint64_t)x[i];

            if (i + 1 == n / 2) {
                oss << "    |";
            }
        }
        return oss.str();
    }
};
