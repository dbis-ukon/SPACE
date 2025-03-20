cmake_minimum_required(VERSION 3.25)
project(PathEnumeration)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(Boost_INCLUDE_DIR /opt/homebrew/Cellar/boost/1.85.0/include/)
set(Boost_LIBRARY_DIR /opt/homebrew/Cellar/boost/1.85.0/lib)

include_directories(${Boost_INCLUDE_DIR})

add_executable(sampler sampler-main.cpp model.cpp model.hpp enumerator.cpp enumerator.hpp trie.cpp trie.hpp
        sampler.cpp
        sampler.hpp)

add_executable(enumerator enumerator-main.cpp model.cpp model.hpp enumerator.cpp enumerator.hpp trie.cpp trie.hpp
        sampler.cpp
        sampler.hpp)
