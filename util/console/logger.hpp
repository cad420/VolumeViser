#pragma once
#include "../common.hpp"
#include <spdlog/spdlog.h>

VUTIL_BEGIN

#define LOG_TRACE(str, ...) spdlog::trace(str, ##__VA_ARGS__)
#define LOG_DEBUG(str, ...) spdlog::debug(str, ##__VA_ARGS__)
#define LOG_INFO(str, ...) spdlog::info(str, ##__VA_ARGS__)
#define LOG_WARN(str, ...) spdlog::info(str, ##__VA_ARGS__)
#define LOG_ERROR(str, ...) spdlog::error(str, ##__VA_ARGS__)
#define LOG_CRITICAL(str, ...) spdlog::critical(str, ##__VA_ARGS__)

#define SET_LOG_LEVEL_TRACE spdlog::set_level(spdlog::level::trace);

#define SET_LOG_LEVEL_DEBUG spdlog::set_level(spdlog::level::debug);

#define SET_LOG_LEVEL_INFO spdlog::set_level(spdlog::level::info);

#define SET_LOG_LEVEL_WARN spdlog::set_level(spdlog::level::warn);

#define SET_LOG_LEVEL_ERROR spdlog::set_level(spdlog::level::err);

#define SET_LOG_LEVEL_CRITICAL spdlog::set_level(spdlog::level::critical);

#define NOT_IMPL LOG_ERROR("this method is not imply yet!");

VUTIL_END