from .util import *

class LZ77:
    def decompress(data):
        xlen = to_uint(data, 0, 4) >> 8

        dest = bytearray(b"\0" * xlen)

        xin = 4
        xout = 0

        while xlen > 0:
            d = to_uint(data, xin, 1)
            xin += 1

            for i in range(8):
                if d & 0x80 == 0x80:
                    stuff = to_uint(data, xin, 1) << 8 | to_uint(data, xin + 1, 1)
                    xin += 2

                    len_ = (stuff >> 12) + 3
                    offset = stuff & 0xfff
                    window_offset = xout - offset - 1

                    for j in range(len_):
                        if window_offset >= len(dest):
                            return dest

                        dest[xout] = dest[window_offset]
                        xout += 1
                        window_offset += 1

                        xlen -= 1
                        if xlen == 0:
                            return dest

                else:
                    dest[xout] = data[xin]
                    xout += 1
                    xin += 1

                    xlen -= 1
                    if xlen == 0:
                        return dest

                d <<= 1

        return dest
