#pragma once

#include <memory>

class VolAnnotaterPrivate;
class VolAnnotater final{
public:
    struct VolAnnotaterCreateInfo{

    };

    explicit VolAnnotater(const VolAnnotaterCreateInfo& info);

    ~VolAnnotater();

    void run();

private:
    std::unique_ptr<VolAnnotaterPrivate> _;
};