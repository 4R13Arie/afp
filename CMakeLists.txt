cmake_minimum_required(VERSION 3.20) # 3.31 is fine too; 3.20 keeps more envs happy
project(afp LANGUAGES CXX)

# -------------------------------------------------------
# Global toolchain / options
# -------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Warnings & basic hygiene
if (MSVC)
    add_compile_options(/W4 /permissive-)
else ()
    add_compile_options(-Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion)
endif ()

# Sanitizers (opt-in)
option(AFP_USE_ASAN "Enable AddressSanitizer" OFF)
option(AFP_USE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
option(AFP_USE_TSAN "Enable ThreadSanitizer" OFF) # NOTE: don't mix with ASAN
function(afp_apply_sanitizers target)
    if (AFP_USE_ASAN)
        target_compile_options(${target} PRIVATE -fsanitize=address -fno-omit-frame-pointer)
        target_link_options(${target} PRIVATE -fsanitize=address)
    endif ()
    if (AFP_USE_UBSAN)
        target_compile_options(${target} PRIVATE -fsanitize=undefined -fno-omit-frame-pointer)
        target_link_options(${target} PRIVATE -fsanitize=undefined)
    endif ()
    if (AFP_USE_TSAN)
        target_compile_options(${target} PRIVATE -fsanitize=thread -fno-omit-frame-pointer)
        target_link_options(${target} PRIVATE -fsanitize=thread)
    endif ()
endfunction()

# -------------------------------------------------------
# Dependencies (via vcpkg recommended)
# -------------------------------------------------------
# tl-expected
find_package(tl-expected CONFIG REQUIRED)           # provides tl::expected

# KFR (DSP)
find_package(KFR CONFIG REQUIRED)                   # provides kfr, kfr_io, kfr_dsp, kfr_dft

# LMDB
find_package(unofficial-lmdb CONFIG REQUIRED)       # provides unofficial::lmdb::lmdb

# GoogleTest for tests (optional unless BUILD_TESTING)
include(CTest)  # sets BUILD_TESTING
if (BUILD_TESTING)
    find_package(GTest CONFIG REQUIRED)               # provides GTest::gtest, GTest::gtest_main
endif ()

# -------------------------------------------------------
# Library target
# -------------------------------------------------------
add_library(afp
        src/afp/dsp/dsp_kfr.cpp
        src/afp/features/features_kfr.cpp
        src/afp/fingerprint/fingerprint_kfr.cpp
        src/afp/index/index_lmdb.cpp
        src/afp/io/io_dr.cpp
        src/afp/match/match_default.cpp
        src/afp/peaks/peaks_kfr.cpp
        src/afp/stft/stft_kfr.cpp
        src/afp/util/util.cpp
        src/afp/all.cpp
)

find_path(VCPKG_INCLUDE_DIR NAMES gtest PATH_SUFFIXES include)

# Public headers live in include/, local single-header libs in lib/
target_include_directories(afp
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${VCPKG_INCLUDE_DIR}
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

target_link_libraries(afp
        PRIVATE
        tl::expected
        kfr
        kfr_io
        kfr_dsp
        kfr_dft
        unofficial::lmdb::lmdb
)

# Apply sanitizers if requested
afp_apply_sanitizers(afp)

# -------------------------------------------------------
# Example executable (optional)
# -------------------------------------------------------
add_executable(afp_exe main.cpp)
target_link_libraries(afp_exe PRIVATE afp)
afp_apply_sanitizers(afp_exe)

# -------------------------------------------------------
# Tests
# -------------------------------------------------------
if (BUILD_TESTING)
    # Test sources (split per module as suggested)
    set(AFP_TEST_SOURCES
            tests/dsp/test_dsp_factory.cpp
            tests/dsp/test_dsp_highpass.cpp
            tests/dsp/test_dsp_lowpass.cpp
            tests/dsp/test_dsp_preemphasis.cpp
            tests/dsp/test_dsp_resampler.cpp

            tests/io/test_utils_io.hpp
            tests/io/test_downmixer.cpp
            tests/io/test_normalize.cpp
            tests/io/test_wav.cpp
            tests/io/test_mp3.cpp
            tests/io/test_flac.cpp
            tests/io/test_composite.cpp
            tests/io/test_edgecases.cpp
    )

    add_executable(afp_tests ${AFP_TEST_SOURCES})
    target_link_libraries(afp_tests PRIVATE afp GTest::gtest GTest::gtest_main)
    target_compile_features(afp_tests PRIVATE cxx_std_20)

    # Point tests at assets dir (if you’re bundling tiny MP3/FLAC samples)
    set(TEST_ASSETS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests/assets)
    target_compile_definitions(afp_tests PRIVATE TEST_ASSETS_DIR="${TEST_ASSETS_DIR}")

    # Copy assets next to test binary (optional)
    add_custom_command(
            TARGET afp_tests POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:afp_tests>/assets
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${TEST_ASSETS_DIR} $<TARGET_FILE_DIR:afp_tests>/assets
    )

    afp_apply_sanitizers(afp_tests)

    include(GoogleTest)
    gtest_discover_tests(afp_tests)
endif ()
