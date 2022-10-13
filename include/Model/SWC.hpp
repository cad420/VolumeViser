#pragma once

#include <IO/SWCIO.hpp>

VISER_BEGIN

//每个节点都有唯一的编号，为了支持随机的删除和插入，
//考虑使用list+map来实现
class SWC{
public:

    struct Point{
        // pos
        // radius
        float radius = 0.f;
        // parent
        Point* prev = nullptr;

    };
    // 减少点的密度，针对算法自动生成的点

    SWC();

    SWC(SWCFile);

    SWCFile ExportToFile() const;

private:
//    std::vector<Point> points;

    std::list<Point> points;
    using Iterator = std::list<Point>::iterator;
    std::unordered_map<int, Iterator> mp;

};

VISER_END

