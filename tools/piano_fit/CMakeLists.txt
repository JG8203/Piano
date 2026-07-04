cmake_minimum_required(VERSION 3.24.0)

juce_add_console_app(PianoFit
    PRODUCT_NAME "Piano Fit"
    VERSION 1.0.0
)

target_sources(PianoFit PRIVATE
    main.cpp
    ${CMAKE_SOURCE_DIR}/plugin/Source/qiano.cpp
    ${CMAKE_SOURCE_DIR}/plugin/Source/dwgs.cpp
    ${CMAKE_SOURCE_DIR}/plugin/Source/hammer.cpp
    ${CMAKE_SOURCE_DIR}/plugin/Source/filter.cpp
    ${CMAKE_SOURCE_DIR}/plugin/Source/reverb.cpp
    ${CMAKE_SOURCE_DIR}/plugin/Source/utils.cpp
    ${CMAKE_SOURCE_DIR}/plugin/Source/AudioFFT.cpp
    ${CMAKE_SOURCE_DIR}/plugin/Source/FFTConvolver.cpp
    ${CMAKE_SOURCE_DIR}/plugin/Source/Utilities.cpp
)

target_include_directories(PianoFit PRIVATE
    ${CMAKE_SOURCE_DIR}/plugin/Source
    ${CMAKE_SOURCE_DIR}/modules/fmt/include
)

target_link_libraries(PianoFit PRIVATE
    gin
    gin_dsp
    gin_simd

    juce::juce_audio_basics
    juce::juce_audio_formats
    juce::juce_core
    juce::juce_data_structures
    juce::juce_events
    juce::juce_graphics
    juce::juce_gui_basics
)

target_compile_definitions(PianoFit PRIVATE
    JUCE_WEB_BROWSER=0
    JUCE_USE_FLAC=0
    JUCE_USE_CURL=0
    JUCE_USE_MP3AUDIOFORMAT=0
    JUCE_USE_LAME_AUDIO_FORMAT=0
    JUCE_USE_WINDOWS_MEDIA_FORMAT=0
    DONT_SET_USING_JUCE_NAMESPACE=1
)

juce_generate_juce_header(PianoFit)

if(APPLE)
    target_compile_options(PianoFit PRIVATE -fno-aligned-allocation)
endif()

if(WIN32)
    target_compile_definitions(PianoFit PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
