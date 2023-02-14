#pragma once

#include "GPUPageTableMgr.hpp"
#include "GPUMemMgr.hpp"
#include "HostMemMgr.hpp"

VISER_BEGIN

class HashPageTable{
public:
    static constexpr int HashTableSize = 1024;
    using HashTableItem = GPUPageTableMgr::PageTableItem;
    using HashTableKey = HashTableItem::first_type;
    using HashTableValue = HashTableItem::second_type;
    inline static HashTableKey INVALID_KEY = GPUPageTableMgr::INVALID_KEY;
    inline static HashTableValue INVALID_VALUE = GPUPageTableMgr::INVALID_VALUE;



    HashPageTable(Ref<GPUMemMgr> gpu_mem_mgr, Ref<HostMemMgr> host_mem_mgr){
        ctx = gpu_mem_mgr->_get_cuda_context();
        hash_page_table = gpu_mem_mgr->AllocBuffer(RescAccess::Shared, HashTableSize * sizeof(HashTableItem));
        hhpt = host_mem_mgr->AllocPinnedHostMem(RescAccess::Shared, HashTableSize * sizeof(HashTableItem), false);
        Clear();
        dirty = false;
    }


    uint32_t GetHashValue(const HashTableKey& key){
        static_assert(sizeof(HashTableKey) == sizeof(int) * 4, "");
        auto p = reinterpret_cast<const uint32_t*>(&key);
        uint32_t v = p[0];
        for(int i = 1; i < 4; i++){
            v = v ^ (p[i] + 0x9e3779b9 + (v << 6) + (v >> 2));
        }
        return v;
    }
    void Update(const HashTableItem& item){
        uint32_t hash_v = GetHashValue(item.first);
        auto pos = hash_v % HashTableSize;
        int i = 0;
        bool positive = false;
        auto table = hhpt->view_1d<HashTableItem>(HashTableSize);
        while(true){
            int ii = i * i;
            pos += positive ? ii : -ii;
            pos %= HashTableSize;
            if(table.at(pos).first == item.first){
                table.at(pos).second = item.second;
                break;
            }
            if(positive)
                ++i;
            positive = !positive;
            if(i >= HashTableSize){
                throw std::runtime_error("HashTable Get Full");
            }
        }
        dirty = true;
//        for(int i = 0; i < HashTableSize; i++){
//            auto& item = table.at(i);
//            std::cout << table.at(i).second.sx << " "
//                      << table.at(i).second.sy << " "
//                      << table.at(i).second.sz << " "
//                      << table.at(i).second.tid << " "
//                      << table.at(i).second.flag
//                      << std::endl;
//        }
    }
    void Append(const HashTableItem& item){
        uint32_t hash_v = GetHashValue(item.first);
        auto pos = hash_v % HashTableSize;
        int i = 0;
        bool positive = false;
        auto table = hhpt->view_1d<HashTableItem>(HashTableSize);
        while(true){
            int ii = i * i;
            pos += positive ? ii : -ii;
            pos %= HashTableSize;
            if(table.at(pos).first == INVALID_KEY){
                table.at(pos) = item;
                break;
            }
            if(positive)
                ++i;
            positive = !positive;
            if(i >= HashTableSize){
                throw std::runtime_error("HashTable Get Full");
            }
        }
        dirty = true;
    }

    void Clear(){
        LOG_DEBUG("call hash table clear");
        auto table = hhpt->view_1d<HashTableItem>(HashTableSize);
        for(int i = 0; i < HashTableSize; i++){
            table.at(i) = {INVALID_KEY, INVALID_VALUE};
        }
        dirty = true;
    }


    Handle<CUDABuffer> GetHandle() {
        if(dirty) Update();
        return hash_page_table;
    }
    //host memory, decide how to transfer by user
    Handle<CUDABuffer> GetHostHandle(){
        return hhpt;
    }

    //将pt从gpu拷贝回host
    void DownLoad(){
        static cub::memory_transfer_info info{
                .width_bytes = HashTableSize * sizeof(HashTableItem),
                .height = 1,
                .depth = 1
        };

        cub::cu_memory_transfer(hash_page_table->view_1d<HashTableItem>(HashTableSize),
                                hhpt->view_1d<HashTableItem>(HashTableSize), info).launch(cub::cu_stream::null(ctx));


    }

    std::vector<HashTableKey> GetKeys(uint32_t flags){
        std::vector<HashTableKey> ret;
        auto table = hhpt->view_1d<HashTableItem>(HashTableSize);
        for(int i = 0; i < HashTableSize; i++){
            if(table.at(i).first != INVALID_KEY){
                const auto& val = table.at(i).second;
                if(val.flag == flags){
                    ret.push_back(table.at(i).first);
                }
            }
        }
        return ret;
    }

private:
    //暂时使用null stream上传，因为数据量很小
    void Update(){
        static cub::memory_transfer_info info{
            .width_bytes = HashTableSize * sizeof(HashTableItem),
            .height = 1,
            .depth = 1
        };
//        auto table = hhpt->view_1d<HashTableItem>(HashTableSize);
//        for(int i = 0; i < HashTableSize; i++){
//            auto& item = table.at(i);
//            std::cout << table.at(i).second.sx << " "
//                      << table.at(i).second.sy << " "
//                      << table.at(i).second.sz << " "
//                      << table.at(i).second.tid << " "
//                      << table.at(i).second.flag
//                      << std::endl;
//        }
        cub::cu_memory_transfer(hhpt->view_1d<HashTableItem>(HashTableSize),
                hash_page_table->view_1d<HashTableItem>(HashTableSize), info).launch(cub::cu_stream::null(ctx));

        dirty = false;
    }
private:
    CUDAContext ctx;
    bool dirty = false;
    Handle<CUDAHostBuffer> hhpt;
    Handle<CUDABuffer> hash_page_table;
};


VISER_END