#pragma once

#include "GPUPageTableMgr.hpp"


VISER_BEGIN

class HashPageTable{
public:
    static constexpr int HashTableSize = 1024;
    using HashTableItem = GPUPageTableMgr::PageTableItem;
    using HashTableKey = HashTableItem::first_type;
    using HashTableValue = HashTableItem::second_type;
    inline static HashTableKey INVALID_KEY = GPUPageTableMgr::INVALID_KEY;
    inline static HashTableValue INVALID_VALUE = GPUPageTableMgr::INVALID_VALUE;

    HashPageTable(){
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

private:
    void Update(){
        //todo transfer to device buffer

        dirty = false;
    }
private:
    bool dirty = false;
    Handle<CUDAHostBuffer> hhpt;
    Handle<CUDABuffer> hash_page_table;
};


VISER_END