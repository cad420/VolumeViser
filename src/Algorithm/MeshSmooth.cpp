#include <Algorithm/MeshSmooth.hpp>

VISER_BEGIN

namespace {

    struct TriFace {
        uint32_t a;
        uint32_t b;
        uint32_t c;

        bool valid() const noexcept {
            return a != INVALID_INDEX && b != INVALID_INDEX && c != INVALID_INDEX;
        }
    };

    struct Node {
        TriFace tri;
        Node *next = nullptr;
    };

    void gaussian_smooth(std::vector<Vertex>& vertices, float c, const std::vector<Node*> neighborhood,
                         std::vector<Float3>& delta_p_list,
                         vutil::thread_group_t& thread_group){
        assert(vertices.size() == neighborhood.size());
        int vert_count = vertices.size();

        auto calc_delta_p = [&](int i, Node* node){
            auto t = node;
            Float3 ret;
            int cnt = 0;
            Float3 vsub;
            while(t){

                const auto& tri = t->tri;
                vsub += vertices[tri.a].pos - vertices[i].pos;
                vsub += vertices[tri.b].pos - vertices[i].pos;
                vsub += vertices[tri.c].pos - vertices[i].pos;

                t =  t->next;
                cnt += 2;
            }
            ret = (1.f / cnt) * vsub;
            return ret;
        };

        vutil::parallel_forrange(0, vert_count, [&](int thread_index, int i){
            delta_p_list[i] = calc_delta_p(i, neighborhood[i]);
        }, thread_group);

        vutil::parallel_forrange(0, vert_count, [&](int thread_index, int i){
            vertices[i].pos += c * delta_p_list[i];
        }, thread_group);

    }

    void regenerate_normal(std::vector<Vertex>& vertices, const std::vector<Node*> neighborhood,
                           vutil::thread_group_t& thread_group){
        assert(vertices.size() == neighborhood.size());
        int vert_count = vertices.size();
        auto calc_normal = [&](int i){
            auto t = neighborhood[i];
            Float3 ret;
            while(t){
                const auto& tri = t->tri;
                if(tri.valid()){
                    auto ab = vertices[tri.b].pos - vertices[tri.a].pos;
                    auto ac = vertices[tri.c].pos - vertices[tri.a].pos;
                    auto norm = vutil::cross(ab.normalized(), ac.normalized());
                    if (!norm.is_nan() && !norm.is_zero() && norm.is_finite())
                        ret += norm.normalized();
                }
                t = t->next;
            }
            return ret.is_zero() ? ret : ret.normalized();
        };
        vutil::parallel_forrange(0, vert_count, [&](int thread_index, int i){
            vertices[i].normal = calc_normal(i);
        });
    }
}

void MeshSmoothing(MeshData0& mesh, float lambda, float mu, int iterations, int worker_count){
    vutil::thread_group_t thread_group;
    thread_group.start(vutil::actual_worker_count(worker_count));
    MeshSmoothing(mesh, lambda, mu, iterations, thread_group);
}

void MeshSmoothing(MeshData0& mesh, float lambda, float mu, int iterations, vutil::thread_group_t& threads){
    assert(mesh.indices.size() % 3 == 0);
    int tri_count = mesh.indices.size() / 3;
    std::vector<Node*> neighborhood(mesh.vertices.size(), nullptr);
    for(int i = 0; i < tri_count; i++){
        TriFace tri{mesh.indices[i * 3], mesh.indices[i * 3 + 1], mesh.indices[i * 3 + 2]};
        Node* t;

        t = new Node();
        t->tri = tri;
        t->next = neighborhood[tri.a];
        neighborhood[tri.a] = t;

        t = new Node();
        t->tri = tri;
        t->next = neighborhood[tri.b];
        neighborhood[tri.b] = t;

        t = new Node();
        t->tri = tri;
        t->next = neighborhood[tri.c];
        neighborhood[tri.c] = t;
    }
    std::vector<Float3> delta_p_list(mesh.vertices.size());
    for(int i = 0; i < iterations; ++i){
        gaussian_smooth(mesh.vertices, lambda, neighborhood, delta_p_list, threads);
        gaussian_smooth(mesh.vertices, mu,     neighborhood, delta_p_list, threads);
        LOG_DEBUG("iteration : {}", i);
    }
    //去除退化的三角形
    auto not_a_triangle = [](const Float3& A, const Float3& B, const Float3& C){
        if(A == B || A == C || B == C) return true;
        auto AB = B - A;
        auto AC = C - A;
        auto N = vutil::cross(AB, AC);
        if(N.is_nan() || N.is_zero() || !N.is_finite()) return true;
        return false;
    };
    auto remove_no_triangle = [&](){
        for(int i = 0; i < tri_count; i++){
            auto& ia = mesh.indices[i * 3];
            auto& ib = mesh.indices[i * 3 + 1];
            auto& ic = mesh.indices[i * 3 + 2];
            auto& A = mesh.vertices[ia].pos;
            auto& B = mesh.vertices[ib].pos;
            auto& C = mesh.vertices[ic].pos;
            if(not_a_triangle(A, B, C)){
                ia = ib = ic = INVALID_INDEX;
            }
        }
    };

    regenerate_normal(mesh.vertices, neighborhood, threads);

    remove_no_triangle();

    LOG_INFO("Finish Mesh Smoothing...");
}

VISER_END