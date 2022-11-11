#include <Algorithm/LevelOfDetailPolicy.hpp>
#include <unordered_set>
VISER_BEGIN



    void ComputeIntersectedBlocksWithViewFrustum(std::vector<GridVolume::BlockUID>& blocks,
                                                 const Float3& block_space,
                                                 const UInt3& block_dim,
                                                 const BoundingBox3D& volume_box,
                                                 const Frustum& frustum,
                                                 std::function<int(const BoundingBox3D&)> lod){
        auto box = FrustumToBoundingBox3D(frustum);
        box &= volume_box;
        box = {box.low - volume_box.low, box.high - volume_box.low};
        assert(box.low.x >= 0 && box.low.y >= 0 && box.low.z >= 0);
        if(!(box.high.x > box.low.x && box.high.y > box.low.y && box.high.z > box.low.z)){
            return;
            auto b = FrustumToBoundingBox3D(frustum);

            LOG_DEBUG("frustum box : {} {} {}, {} {} {}", b.low.x, b.low.y, b.low.z, b.high.x, b.high.y, b.high.z);
        }
        auto idx_beg = UInt3(box.low / block_space);
        auto idx_end = UInt3(box.high/ block_space) + UInt3(1);
        idx_end.x = std::min(idx_end.x, block_dim.x);
        idx_end.y = std::min(idx_end.y, block_dim.y);
        idx_end.z = std::min(idx_end.z, block_dim.z);
        std::vector<std::pair<GridVolume::BlockUID, BoundingBox3D>> lod0_blocks;
        for(auto z = idx_beg.z; z < idx_end.z; z++){
            for(auto y = idx_beg.y; y < idx_end.y; y++){
                for(auto x = idx_beg.x; x < idx_end.x; x++){
                    GridVolume::BlockUID block_uid = {x, y, z, 0};
                    BoundingBox3D block_box = {{
                        volume_box.low.x + x * block_space.x,
                        volume_box.low.y + y * block_space.y,
                        volume_box.low.z + z * block_space.z
                        },{
                        volume_box.low.x + (x + 1) * block_space.x,
                        volume_box.low.y + (y + 1) * block_space.y,
                        volume_box.low.z + (z + 1) * block_space.z
                    }};
                    if(GetBoxVisibility(frustum, block_box) != BoxVisibility::Invisible){
                        lod0_blocks.emplace_back(block_uid, block_box);
                    }
                    else{
//                        LOG_DEBUG("invisible block : {} {} {}",block_uid.x, block_uid.y, block_uid.z);
                    }
                }
            }
        }
        LOG_DEBUG("Lod0 intersect blocks count : {}", lod0_blocks.size());
        std::unordered_set<GridVolume::BlockUID> res;
        for(auto& b : lod0_blocks){
            uint32_t l = lod(b.second);
            UInt3 idx = {b.first.x, b.first.y, b.first.z};
            for(int i = 0; i < l; ++i){
                idx = idx / 2u;
            }
            res.insert(GridVolume::BlockUID{idx.x, idx.y, idx.z, l});
        }
        blocks.reserve(res.size());
        for(auto& b : res){
            blocks.emplace_back(b);
        }

    }

    template<typename ViewSpace>
    void ComputeIntersectedBlocksWithViewSpace(std::vector<GridVolume::BlockUID>& blocks,
                                                 const GridVolume& volume,
                                                 const ViewSpace& space,
                                                 const Float3& block){

    }

    void ComputeUpBoundLOD(LevelOfDist& lods, float base_space, int width, int height, float fov){
        float sampling_space = 0.5f * base_space;
        //中心像素对应的光线夹角最大，距离上限最小
        for(int lod = 0; lod < LevelOfDist::MaxLevelCount; lod++){
            float space = sampling_space * (1 << lod);
            float dy = height * space / (2.f * tan(0.5f * fov));
            lods.LOD[lod] = dy;
        }
    }
    void ComputeDefaultLOD(LevelOfDist& lods, Float3 block_space){
        auto base_lod_dist = block_space.length() * 0.5f;
        for(int lod = 0; lod < LevelOfDist::MaxLevelCount; lod++){
            lods.LOD[lod] = base_lod_dist * (1 << lod);
        }
        lods.LOD[LevelOfDist::MaxLevelCount - 1] = std::numeric_limits<float>::max();
    }

    template<>
    void ComputeIntersectedBlocksWithViewSpace<Frustum>(std::vector<GridVolume::BlockUID>& blocks,
                                                 const GridVolume& volume,
                                                 const Frustum & space,
                                                 const Float3& block);

    template<>
    void ComputeIntersectedBlocksWithViewSpace<BoundingBox3D>(std::vector<GridVolume::BlockUID>& blocks,
                                                        const GridVolume& volume,
                                                        const BoundingBox3D& space,
                                                        const Float3& block);

VISER_END