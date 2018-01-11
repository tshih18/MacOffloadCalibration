# -*- mode: python -*-

block_cipher = None


a = Analysis(['MacHomeCalServerGuiClass.py'],
             pathex=['/Users/theodoreshih/Desktop/Work/OffloadCalibration/MacOffloadCalibration'],
             binaries=[],
             datas=[('dist/saveme1', '.'), ('dist/mul_proc_offsetv4', '.')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)

a.datas += [('psi_icon.png','/Users/theodoreshih/Desktop/Work/OffloadCalibration/MacOffloadCalibration/psi_icon.png', '.'),
            ('offset_icon.png','/Users/theodoreshih/Desktop/Work/OffloadCalibration/MacOffloadCalibration/offset_icon.png', '.')]

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='MacHomeCalServerGuiClass',
          debug=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=False )
app = BUNDLE(exe,
             name='MacHomeCalServerGuiClass.app',
             icon=None,
             bundle_identifier=None)
