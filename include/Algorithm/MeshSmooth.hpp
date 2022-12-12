#pragma once

#include <Common/Common.hpp>
#include <IO/MeshIO.hpp>
VISER_BEGIN

void MeshSmoothing(MeshData0& mesh, float lambda, float mu, int iterations, int worker_count = 0);

void MeshSmoothing(MeshData0& mesh, float lambda, float mu, int iterations, vutil::thread_group_t& threads);

VISER_END