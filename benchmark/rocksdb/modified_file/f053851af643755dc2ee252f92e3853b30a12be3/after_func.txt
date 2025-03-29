void CompactionPicker::GetGrandparents(
    VersionStorageInfo* vstorage, const CompactionInputFiles& inputs,
    const CompactionInputFiles& output_level_inputs,
    std::vector<FileMetaData*>* grandparents) {
  InternalKey start, limit;
  GetRange(inputs, output_level_inputs, &start, &limit);
  // Compute the set of grandparent files that overlap this compaction
  // (parent == level+1; grandparent == level+2 or the first
  // level after that has overlapping files)
  for (int level = output_level_inputs.level + 1; level < NumberLevels();
       level++) {
    vstorage->GetOverlappingInputs(level, &start, &limit, grandparents);
    if (!grandparents->empty()) {
      break;
    }
  }
}
