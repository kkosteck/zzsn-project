# ZZSN Projekt - instrukcja obsługi

W celu zainstalowania wszelkich potrzebnych biblioteki i modułów konieczne jest użycie komendy:
```
conda env create -f environment.yml
```
## Uruchomienie treningu oraz testowania modelu
Wszelkie funkcje ubsługujące trening oraz testowanie zaimplementowanego modelu znajdują się w pliku Jupyter Notebook o nazwie: 

[train_SNN.ipynb](https://github.com/kkosteck/zzsn-project/blob/master/train_SNN.ipynb)

Parametry modelu można ustawić bezpośredni w pliku Notebook lub w pliku konfiguracyjnym: [config.json](https://github.com/kkosteck/zzsn-project/blob/master/config.json) - poszczególne właściwości parametrów są opisane bezpośrednio w Notebooku.

Sam przebieg treningu oraz testów modelu przebiega bezpośrednio w notebooku train_SNN.ipynb - wszelkie wartości użytych parametrów zostały opisane w dokumnetacji.
## Przykłady z biblioteki BindsNet
Użyte przez nas przykłady stworzone przez autorów frameworka BindsNet można znaleźć w folderze [examples](https://github.com/kkosteck/zzsn-project/tree/master/examples) - dokładna instrukcja uruchomienia jest opisana bezpośrednio w bibliotece [BindsNet](https://github.com/BindsNET/bindsnet/tree/master/examples)
