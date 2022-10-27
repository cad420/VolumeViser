#include <Algorithm/LevelOfDetailPolicy.hpp>

VISER_BEGIN



    void ComputeIntersectedBlocksWithViewFrustum(std::vector<GridVolume::BlockUID>& blocks,
                                                 const GridVolume& volume,
                                                 const Frustum & space,
                                                 const Float3& view,
                                                 std::function<int(float)> lod){

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