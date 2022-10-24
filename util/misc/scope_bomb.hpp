#pragma once

#include "concept.hpp"
#include <type_traits>

VUTIL_BEGIN

    template <typename T,typename = std::enable_if_t<std::is_invocable_v<T>>>
    class scope_bomb_t:public no_copy_t,no_heap_t{
        bool should_call = true;
        T bomb;
    public:
        explicit scope_bomb_t(T&& t)
                :bomb(std::forward<T>(t))
        {}
        ~scope_bomb_t(){
            if(should_call && bomb)
                bomb();
        }
        void dismiss(){
            should_call = false;
        }
        void call(){
            assert(should_call && bomb);
            bomb();
            should_call = false;
        }
    };

VUTIL_END