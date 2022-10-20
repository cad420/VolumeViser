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

    void ComputeUpBoundLOD(LevelOfDist& lod, int width, int height, float fov){

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