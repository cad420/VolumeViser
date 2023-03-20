#include <Core/RenderParams.hpp>

VISER_BEGIN

    namespace{
    template<typename Key, typename Value>
    void Gen1DTFImpl(const TransferFuncT<Key, Value>& tf, Value* ptr, int dim){
        if(tf.pts.empty()) return;
        std::map<Key, Value> tf_mp_pts;
        std::memset(ptr, 0, sizeof(Value) * dim);
        double dt = 1.0 / dim;
        bool first = true;
        int last = -1;
        // first and last key and point
        tf_mp_pts[0] = tf.pts.begin()->second;
        tf_mp_pts[1.f] = std::prev(tf.pts.end())->second;
        for(auto& [k, v] : tf.pts) tf_mp_pts[k] = v;
        for(auto& pt : tf_mp_pts){
            int cur = std::max<int>(0, std::floor<int>(static_cast<int>(pt.first * dim) - 0.5));
            ptr[cur] = pt.second;
            if(first){
                last = cur;
                first = false;
                continue;
            }
            int d = cur - last;
            for(int i = last + 1; i < cur; i++){
                auto v = ptr[last], vv = ptr[cur];
                ptr[i] = ptr[last] * static_cast<Key>(cur - i) / static_cast<Key>(d)
                        + ptr[cur] * static_cast<Key>(i - last) / static_cast<Key>(d);
//                std::cout << "i: " << i << ", " << ptr[i] << std::endl;
            }
            last = cur;
        }
    }

    template<typename Key, typename Value>
    void Gen2DTFImpl(const TransferFuncT<Key, Value>& tf, const Value* tf1d, Value* ptr, int dim){
        const int base_sampler_number = 20;
        const int ratio = 1;

        const Key ray_step = 1.0;
        for(int sb = 0; sb < dim; sb++){
            for(int sf = 0; sf <= sb; sf++){
                int offset = sb != sf;
                int n = base_sampler_number + ratio * std::abs(sb - sf);
                Key step_width = ray_step / n;
                Value rgba;
                for(int i = 0; i < n; i++){
                    Key s = sf + static_cast<Key>(sb - sf) * i / n;
                    float s_frac = s - std::floor(s);
                    float opacity = (tf1d[int(s)].w * (Key(1.0) - s_frac)
                            + tf1d[(int)s + offset].w * s_frac) * step_width;
                    float tmp = std::exp(-rgba.w) * opacity;

                }
            }
        }
    }
}

template<typename Key, typename Value>
std::vector<Value> TransferFuncT<Key, Value>::Gen1DTF(int dim) const {
    std::vector<Value> ret(dim);
    Gen1DTFImpl<Key,Value>(*this, ret.data(), dim);
    return ret;
}

template<typename Key, typename Value>
std::vector<Value> TransferFuncT<Key, Value>::Gen2DTF(const std::vector<Value>& tf1d, int dim) const {
    std::vector<Value> ret(dim);
    Gen2DTFImpl<Key, Value>(*this, tf1d.data(), ret.data(), dim);
    return ret;
}

template<typename Key, typename Value>
void TransferFuncT<Key, Value>::Gen1DTF(Handle<CUDAHostBuffer> &buffer, int dim) const {
    auto lk = buffer.AutoLocker();
    Gen1DTFImpl(*this, reinterpret_cast<Value*>(buffer->get_data()), dim);
}

template<typename Key, typename Value>
void TransferFuncT<Key, Value>::Gen2DTF(Handle<CUDAHostBuffer> & tf1d, Handle<CUDAHostBuffer> &buffer, int dim) const {
    auto lk = buffer.AutoLocker();
    Gen2DTFImpl(*this, reinterpret_cast<const Value*>(tf1d->get_data()), reinterpret_cast<Value*>(buffer->get_data()), dim);
}

template struct TransferFuncT<float, Float4>;

VISER_END