#!/usr/bin/env bash

# in adition to your IDE clang-format,it is worth running this
# periodically to ensure all formatting errors have been caught.

find test/ include/ bench/ -type f -name '*.cpp' -or -name '*.hpp' | xargs clang-format --verbose -i
