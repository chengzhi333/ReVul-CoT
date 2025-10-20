# coding:utf-8
from tree_sitter import Language

Language.build_library(

  # Store the library in the `build` directory
  'build/my-languages.so',

  # Include one or more languages
  [
    'D:/python project/pythonProject/RAG-LLM/treesitter/tree-sitter-c-master/tree-sitter-c-master',
    'D:/python project/pythonProject/RAG-LLM/treesitter/tree-sitter-cpp-master/tree-sitter-cpp-master'
    # 'treesitter/tree-sitter-java',
    # 'treesitter/tree-sitter-python',
    # 'treesitter/tree-sitter-cpp',
  ]
)