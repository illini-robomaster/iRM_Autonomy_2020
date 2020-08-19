add_subdirectory(lcm-1.4.0)
add_subdirectory(googletest-1.10.0)

# suppress LCM build warnings
target_compile_options(lcm PRIVATE -Wno-deprecated-declarations "-Wno-#pragma-messages")
target_compile_options(lcm-static PRIVATE -Wno-deprecated-declarations "-Wno-#pragma-messages")
target_compile_options(lcm-gen PRIVATE -Wno-deprecated-declarations -Wno-format)
target_compile_options(lcm-logger PRIVATE
    -Wno-deprecated-declarations
    $<$<PLATFORM_ID:UNIX>: -Wno-format-truncation>
    "-Wno-#pragma-messages"
    -Wno-implicit-function-declaration)
