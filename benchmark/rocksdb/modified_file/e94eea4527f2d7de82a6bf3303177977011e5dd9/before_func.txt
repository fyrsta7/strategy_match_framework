void WriteBatch::Clear() {
  rep_.clear();
  rep_.resize(kHeader);
}
