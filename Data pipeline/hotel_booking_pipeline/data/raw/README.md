# Original paper datasets

來源：[Hotel booking demand datasets](https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/) Appendix A 的 `mmc2.zip`。此壓縮檔同時包含 `H1.csv` 與 `H2.csv`，已原樣解出，未清理或合併。

| File | Hotel | Bookings | SHA-256 |
| --- | --- | ---: | --- |
| `H1.csv` | Resort Hotel | 40,060 | `6F38FDA13794FD6CF6547D2432FFB79666200000EB45FD09E4B97FE49519A6AB` |
| `H2.csv` | City Hotel | 79,330 | `5107090FFDC5585A9E18299801E9D85C3B459C235542882183A502A73EF63C4D` |
| `mmc2.zip` | Original supplement | — | `6DB57D635B9EDEC05F1CF9616BA6473E864909E4FC34819FA21CFF58D518EE6A` |

兩個 CSV 各有 31 個原始欄位，未修改內容或列順序。交付包含 H1/H2 CSV；上表的 ZIP 雜湊僅供來源核對，執行 pipeline 不需要 ZIP 或其他資料夾。H1/H2 分開處理，Booking_ID 依原始列號產生，因此不要重排原始 CSV。
