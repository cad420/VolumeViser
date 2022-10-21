#pragma once

#include "GPUPageTableMgr.hpp"


VISER_BEGIN

class HashPageTable{
public:
    static constexpr int HashTableSize = 1024;
    using HashTableItem = GPUPageTableMgr::PageTableItem;
    using HashTableKey = HashTableItem::first_type;

    uint32_t GetHashValue(const HashTableKey& key){

    }

    void Append(const HashTableItem& item){

    }

    void Clear(){

    }

    void Lock(){

    }

    void UnLock(){

    }

    Handle<CUDABuffer> GetHandle() {
        return hash_page_table;
    }

private:
    Handle<CUDABuffer> hash_page_table;
};


VISER_END