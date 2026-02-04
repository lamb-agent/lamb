#! /bin/bash

ollama serve &
ollama pull llama4:16x17b
pkill ollama
