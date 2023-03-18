#pragma once


#include "Common.hpp"

class NeuronRenderer{
  public:
    struct NeuronRendererCreateInfo{
        ID3D11Device* dev = nullptr;

        int frame_w = 0, frame_h = 0;

    };

    void Render();

    ComPtr<ID3D11ShaderResourceView> GetSRV();



  private:

};