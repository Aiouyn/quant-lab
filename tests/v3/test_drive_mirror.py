from desk.data import drive


def test_drive_reads_the_local_mirror_before_rclone(tmp_path, monkeypatch):
    d = tmp_path / 'QUANT_DATA' / 'hyperliquid' / 'asset_ctxs'
    d.mkdir(parents=True)
    (d / '20260101.csv.lz4').write_bytes(b'abc')
    monkeypatch.setenv('DESK_DRIVE_ROOT', str(tmp_path))
    monkeypatch.setattr(drive.subprocess, 'run', lambda *a, **k: (_ for _ in ()).throw(AssertionError('rclone used')))
    assert drive.cat('loopfactory:QUANT_DATA/hyperliquid/asset_ctxs/20260101.csv.lz4') == b'abc'
    assert drive.lsf('loopfactory:QUANT_DATA/hyperliquid/asset_ctxs') == ['20260101.csv.lz4']
