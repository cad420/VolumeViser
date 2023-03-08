#include <Core/HashPageTable.hpp>


VISER_BEGIN

void HashPageTable::DownLoad()
{
    static cub::memory_transfer_info info{
        .width_bytes = HashTableSize * sizeof(HashTableItem),
        .height = 1,
        .depth = 1
    };


    cub::cu_memory_transfer(hash_page_table->view_1d<HashTableItem>(HashTableSize),
                            hhpt->view_1d<HashTableItem>(HashTableSize), info).launch(cub::cu_stream::null(ctx));


}

std::vector<HashPageTable::HashTableKey> HashPageTable::GetKeys(uint32_t flags)
{
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

void HashPageTable::Update()
{

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

VISER_END
