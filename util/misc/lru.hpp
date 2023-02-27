#pragma once

#include "../common.hpp"
#include "../console/logger.hpp"
#include <unordered_map>
#include <list>
#include <optional>

VUTIL_BEGIN

    template<typename Key,typename Value,typename Hash=std::hash<Key>>
    class lru_t
    {
    public:
        using ItemType = std::pair<Key,Value>;
        using ItemIterator = typename std::list<ItemType>::iterator;

        explicit lru_t(size_t cap)
        :capacity(cap)
        {}

        void clear(){
            data.clear();
            pos.clear();
        }

        Value* get_value_ptr(const Key& key){
            auto it = pos.find(key);
            if(it == pos.end())
                return nullptr;
            move_to_head(it->second);
            return &(data.begin()->second);
        }

        std::optional<Value> get_value(const Key& key){
            auto it = pos.find(key);
            if(it == pos.end())
                return std::optional<Value>(std::nullopt);
            move_to_head(it->second);
            return std::make_optional<Value>(data.begin()->second);
        }

        std::optional<Value> front_value() const{
            if(data.size() == 0)
                return std::optional<Value>(std::nullopt);
            return std::make_optional<Value>(data.begin()->second);
        }

        ItemType& back() {
            assert(data.size());
            return data.back();
        }

        /**
         * if key exists then the value of key will update and move this item to head
         */
        void emplace_back(const Key& key,Value value){
            auto it = pos.find(key);
            if(it != pos.end()){
//                LOG_DEBUG("find");
                it->second->second = std::move(value);
                move_to_head(it->second);
                return;
            }
//            LOG_DEBUG("not exist");
            if(data.size() >= capacity){
                pos.erase(data.back().first);//erase by key for unordered_map
                data.pop_back();
            }
//            LOG_DEBUG("1111");
            data.emplace_front(std::make_pair(key,std::move(value)));
//            LOG_DEBUG("2222");
            pos[key] = data.begin();
        }
        float get_load_factor() const{
            return 1.f * data.size() / capacity;
        }
    private:
        void move_to_head(ItemIterator& it){
            auto key = it->first;
            data.emplace_front(std::move(*it));
            data.erase(it);
            pos[key] = data.begin();
        }
    private:
        std::unordered_map<Key,ItemIterator,Hash> pos;
        std::list<ItemType> data;
        size_t capacity;
    };


VUTIL_END