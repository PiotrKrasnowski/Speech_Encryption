# Speech_Encryption

26/05/2022, Paris, France

UPDATE: the code is currently available at: https://osf.io/gujfp/ .

14/12/2020, Sophia Antipolis, France

Due to storage limitations, recordings have been moved to Google Drive (link below):

https://drive.google.com/drive/folders/1Zla9VbV64JQekpb5GpDtJKMYbWZ7Xw5f?usp=sharing

Upon request, I will be happy to disclose parts of the Python code used in simulations. Contact: krasnowski@i3s.unice.fr

17/02/2020, Sophia Antipolis, France

Parts of the Python code are available now on GitHub. The code performs full enciphering/deciphering chains and simulates channel noise/compression. The result of the processing is a file with input data used by LPCNet: https://github.com/mozilla/LPCNet . Please mind that the input vector of the narrowband LPCNet is slightly modified and should me changed accordingly. Moreover, running the code may require updating some system libraries (used by the speech encoder).

The code uses a modified LPCNet speech encoder from https://github.com/mozilla/LPCNet (an executable file speech_encoder), a pitch detector https://github.com/jkjaer/fastF0Nls, and a publicly avail;able source code of some speech coders (e.g., OPUS-SILK).
