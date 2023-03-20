#pragma once

#include "../common.hpp"

#include <atomic>

VUTIL_BEGIN

    class rw_spinlock_t{
    public:
        enum{
            Write = 1,
            Read = 2
        };

        rw_spinlock_t(){
            counter.store(0, std::memory_order_relaxed);
        }

        // add read lock and wait for write unlock, avoid r/w race
        inline void lock_read(){
            auto v = counter.fetch_add(Read,std::memory_order_acquire);
            while((v & Write) != 0){
                v = counter.load(std::memory_order_acquire);
            }
        }

        inline void unlock_read(){
            counter.fetch_sub(Read,std::memory_order_release);
        }

        // wait for read/write unlock and add write lock
        inline void lock_write(){
            uint32_t expected = 0;
            while(!counter.compare_exchange_weak(expected,Write,std::memory_order_acquire,
                                                 std::memory_order_relaxed)){
                expected = 0;
            }
        }

        inline void unlock_write(){
            counter.fetch_and(~Write,std::memory_order_release);
        }

        inline void unlock_all(){
            counter.store(0, std::memory_order_relaxed);
        }

        inline void converse_read_to_write(){
            uint32_t expected = Read;
            //just consider one read to write
            if(!counter.compare_exchange_strong(expected,Write,
                                                std::memory_order_acquire,
                                                std::memory_order_relaxed)){
                unlock_read();
                lock_write();
            }
        }

        inline void converse_write_to_read(){
            uint32_t expected = Write;
            if(!counter.compare_exchange_strong(expected, Read,
                                                std::memory_order_acquire,
                                                std::memory_order_relaxed)){
                expected = Write;
                unlock_write();
                lock_read();
            }
        }

        bool is_read_locked(){
            auto v = counter.load();
            return v >> 1;
        }

        bool is_write_locked(){
            auto v = counter.load();
            return v & 1;
        }

    private:
        std::atomic_uint32_t counter;
    };

VUTIL_END